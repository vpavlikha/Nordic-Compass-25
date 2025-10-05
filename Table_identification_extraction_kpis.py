# -*- coding: utf-8 -*-
"""
Combo pipeline:
1) Convert all PDFs in Reports/ to page PNGs (300 dpi)
2) Detect table regions on pages with LayoutParser EfficientDet (PubLayNet) -> save crops
3) Run Camelot on PDF pages using LP-detected areas -> HTML/CSV + KPI extraction
Requires:
- python -m pip install layoutparser[ocr] opencv-python pypdfium2 beautifulsoup4 pandas numpy camelot-py
- Ghostscript installed and on PATH for Camelot lattice mode
"""

import os
import re
import json
import traceback
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---- PATHS (edit if you like) ----
BASE_DIR      = Path(r"C:\Users\vikto\Desktop\Nordic Compass")
REPORTS_DIR   = BASE_DIR / "Reports"     # <-- Put PDFs here
PAGES_DIR     = BASE_DIR / "Pages"
CROPS_DIR     = BASE_DIR / "CropsLP"
EXTRACT_ROOT  = BASE_DIR / "Extracted"

# LayoutParser model cache path (download once)
LP_WEIGHT_URL = "https://www.dropbox.com/s/gxy11xkkiwnpgog/publaynet-tf_efficientdet_d1.pth.tar?dl=1"
LP_WEIGHT_LOC = Path(r"C:\Users\vikto\lp_models\publaynet-tf_efficientdet_d1.pth.tar")

# ---- SETTINGS ----
DPI = 300
LP_PADDING    = 15
LP_MIN_WIDTH  = 100
LP_MIN_HEIGHT = 50
SHOW_LOG = False
LANG     = "en"
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Optional page range via environment variables (1-based, inclusive)
ENV_PAGE_START = os.environ.get("PAGE_START")
ENV_PAGE_END   = os.environ.get("PAGE_END")
PAGE_START: Optional[int] = int(ENV_PAGE_START) if (ENV_PAGE_START and ENV_PAGE_START.isdigit()) else None
PAGE_END: Optional[int]   = int(ENV_PAGE_END) if (ENV_PAGE_END and ENV_PAGE_END.isdigit()) else None

# ============ IMPORTS (heavy) ============
import urllib.request
import cv2
import numpy as np
import pandas as pd
import layoutparser as lp
import pypdfium2 as pdfium
from bs4 import BeautifulSoup
import camelot
import shutil

# ---------- Ensure dirs ----------
for d in [REPORTS_DIR, PAGES_DIR, CROPS_DIR, EXTRACT_ROOT, LP_WEIGHT_LOC.parent]:
    d.mkdir(parents=True, exist_ok=True)

# ---------- Detect LP EfficientDet availability ----------
HAS_LP_EFFDET = hasattr(lp, "EfficientDetLayoutModel")

# ---------- Download LP weight once (only if model available) ----------
if HAS_LP_EFFDET:
    if not LP_WEIGHT_LOC.exists():
        print("Downloading LayoutParser model weights...")
        urllib.request.urlretrieve(LP_WEIGHT_URL, str(LP_WEIGHT_LOC))

# ---------- Init LP model (guarded) ----------
lp_model = None
if HAS_LP_EFFDET:
    try:
        print("Initializing LayoutParser model...")
        lp_model = lp.EfficientDetLayoutModel(
            config_path="lp://PubLayNet/tf_efficientdet_d1/config",
            model_path=str(LP_WEIGHT_LOC),
            extra_config={"score_thresh": 0.8},
            enforce_cpu=True,
        )
        print("LayoutParser model initialized successfully")
    except Exception as e:
        print(f"Failed to initialize LayoutParser model: {e}")
        lp_model = None
else:
    print("LayoutParser EfficientDet not available")

# ---------- Camelot settings (no Paddle dependency) ----------
# We will use LayoutParser to detect table regions on the page image,
# convert those pixel bboxes to PDF point coordinates, and pass them
# as Camelot table_areas to extract accurate tables from the original PDF page.

# ====== UTIL: robust read (Windows unicode) ======
def load_bgr_3ch(img_path: str) -> np.ndarray:
    data = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

# ---------- STEP 1: PDF -> pages ----------
def pdf_to_pages(pdf_path: Path, out_dir: Path, dpi: int = DPI) -> List[Path]:
    """Render PDF pages to PNG using pypdfium2 (no Poppler required)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(str(pdf_path))
    num_pages = len(doc)
    scale = float(dpi) / 72.0
    out: List[Path] = []
    try:
        for i in range(num_pages):
            page = doc[i]
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            img_path = out_dir / f"page_{i+1}.png"
            pil_image.save(str(img_path), format="PNG")
            out.append(img_path)
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return out

# ---------- STEP 2: LP table crops ----------
def detect_and_crop_tables(image_path: Path, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  ! Skip (could not read): {image_path}")
        return []

    h, w = image.shape[:2]
    layout = lp_model.detect(image)
    table_blocks = [b for b in layout if b.type == "Table"]

    crops = []
    for i, block in enumerate(table_blocks, start=1):
        x1, y1, x2, y2 = map(int, block.coordinates)
        x1 = max(0, x1 - LP_PADDING); y1 = max(0, y1 - LP_PADDING)
        x2 = min(w, x2 + LP_PADDING); y2 = min(h, y2 + LP_PADDING)
        if (x2 - x1) < LP_MIN_WIDTH or (y2 - y1) < LP_MIN_HEIGHT:
            continue
        crop = image[y1:y2, x1:x2]
        out_path = out_dir / f"{image_path.stem}_table_{i}.png"
        cv2.imwrite(str(out_path), crop)
        crops.append(out_path)
    return crops

# ---------- STEP 3: PP-Structure + parsing ----------
def _clean_html_for_read_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup.find_all(text=True):
        if t.string:
            t.replace_with(
                t.string.replace("\u00ad", " ").replace("\xa0", " ").replace("\u202f", " ")
            )
    return str(soup)

def _try_multiheader_read(html: str) -> List[pd.DataFrame]:
    cleaned = _clean_html_for_read_html(html)
    candidates = []
    for header in ([0], [0,1], [0,1,2]):
        try:
            dfs = pd.read_html(cleaned, header=header)
            for df in dfs:
                if df.shape[0] * df.shape[1] >= 6:
                    candidates.append(df)
        except Exception:
            continue
    return candidates

def _flatten_columns(cols) -> List[str]:
    if not isinstance(cols, pd.MultiIndex):
        return [str(c).strip() for c in cols]
    tuples = [list(map(lambda x: "" if (x is None or str(x).lower()=="nan") else str(x).strip(), tup))
              for tup in cols.to_list()]
    for j in range(1, len(cols.levels)):
        for i, tup in enumerate(tuples):
            if tup[j] == "":
                tup[j] = tup[j-1]
    flat = [" | ".join([p for p in tup if p]) for tup in tuples]
    return [re.sub(r"\s{2,}", " ", f).strip(" |") for f in flat]

def _is_section_row(row: pd.Series, numeric_cols: List[str]) -> bool:
    first = str(row.iloc[0]).strip()
    if not first or first.lower() in ("nan",):
        return False
    if not numeric_cols:
        return False
    for c in numeric_cols:
        val = str(row.get(c, "")).strip()
        if val not in ("", "–", "-", "—", "nan"):
            return False
    return True

def _parse_number(x: str) -> Optional[float]:
    if x is None:
        return np.nan
    s = str(x).strip()
    if s in ("", "-", "–", "—"):
        return np.nan
    s = s.replace("%", "")
    s = s.replace("\u202f", " ").replace("\xa0", " ")
    s = re.sub(r"(?<=\d)[ ,](?=\d{3}\b)", "", s)
    s = s.replace(",", ".")
    if re.fullmatch(r"\(\s*[\d\.]+\s*\)", s):
        s = "-" + re.sub(r"[() ]", "", s)
    try:
        return float(s)
    except Exception:
        return np.nan

def _looks_year_name(name: str) -> Optional[str]:
    m = re.search(r"\b(19|20)\d{2}\b", str(name))
    return m.group(0) if m else None

def html_to_best_df(html_list: List[str]) -> pd.DataFrame:
    cands = []
    for h in html_list:
        cands.extend(_try_multiheader_read(h))
    if not cands:
        raise RuntimeError("Could not parse any table HTML.")
    best = max(cands, key=lambda d: d.shape[0] * d.shape[1])
    best.columns = _flatten_columns(best.columns)
    best = best[(best.astype(str).replace("", np.nan)).notna().any(axis=1)].reset_index(drop=True)
    best = best.applymap(lambda v: v.strip() if isinstance(v, str) else v)
    return best

def build_wide_and_long(df_in: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df_in.copy()
    numeric_likelihood = {}
    for c in df.columns:
        s = df[c].astype(str).head(30)
        nums = s.map(lambda x: _parse_number(x)).notna().mean()
        numeric_likelihood[c] = nums
    item_col = min(numeric_likelihood, key=numeric_likelihood.get)
    if "Item" not in df.columns:
        df.rename(columns={item_col: "Item"}, inplace=True)

    def _note_like(s: pd.Series) -> bool:
        t = s.astype(str).str.replace(",", ".").str.replace(r"\s*-\s*", ".", regex=True).str.replace(r"\s+","", regex=True)
        ok = t.str.fullmatch(r"(?:\d|[1-9]\d?)(?:\.\d{1,2})?").mean()
        return ok >= 0.6

    for c in df.columns:
        if c == "Item":
            continue
        if _note_like(df[c].head(25)):
            df.rename(columns={c: "Note"}, inplace=True)
            break

    year_cols, year_name_map = [], {}
    for c in df.columns:
        if c in ("Item", "Note"):
            continue
        y = _looks_year_name(c)
        if y:
            year_cols.append(c)
            year_name_map[c] = y
    if not year_cols:
        year_cols = [c for c in df.columns if c not in ("Item", "Note")]
    for c in year_cols:
        df[c] = df[c].map(_parse_number)

    numeric_cols_for_section = year_cols.copy()
    section = []
    cur = None
    for _, row in df.iterrows():
        if _is_section_row(row, numeric_cols_for_section):
            cur = str(row["Item"]).strip()
        section.append(cur)
    df.insert(0, "Section", section)

    mask_section_only = df.apply(lambda r: _is_section_row(r, numeric_cols_for_section), axis=1)
    df_wide = df.loc[~mask_section_only].reset_index(drop=True)

    df_wide = df_wide.rename(columns=year_name_map)
    years_in_cols = sorted([c for c in df_wide.columns if re.fullmatch(r"(19|20)\d{2}", str(c))], reverse=True)
    lead = ["Section", "Item"] + (["Note"] if "Note" in df_wide.columns else [])
    others = [c for c in df_wide.columns if c not in lead + years_in_cols]
    df_wide = df_wide[lead + years_in_cols + others]

    if years_in_cols:
        df_long = df_wide.melt(
            id_vars=lead, value_vars=years_in_cols, var_name="Year", value_name="Value"
        ).dropna(subset=["Value"]).reset_index(drop=True)
    else:
        numeric_cols = [c for c in df_wide.columns if c not in lead]
        df_long = df_wide.melt(
            id_vars=lead, value_vars=numeric_cols, var_name="Measure", value_name="Value"
        ).dropna(subset=["Value"]).reset_index(drop=True)

    df_wide["Item"] = (df_wide["Item"].astype(str)
                       .str.replace(r"\.{2,}", "", regex=True)
                       .str.replace(r"\s{2,}", " ", regex=True)
                       .str.strip())
    df_long["Item"] = (df_long["Item"].astype(str)
                       .str.replace(r"\.{2,}", "", regex=True)
                       .str.replace(r"\s{2,}", " ", regex=True)
                       .str.strip())
    return df_wide, df_long

KPI_PATTERNS: Dict[str, Dict] = {
  "Sales (MEUR)": {
    "aliases": ["sales","net sales","revenue","revenues","net revenue","turnover","net turnover","total revenue","group revenue","sales revenue","operating revenue","income from sales"],
    "regex": r"""\b(?:net\s+)?(?:sales|revenue|turnover|group\s+revenue|total\s+revenue)\b(?:[^A-Za-z0-9]+(?:MEUR|EURm|€m|EUR\s*million|mEUR|M€))?(?:[^A-Za-z0-9]+(?:FY|YE|year[-\s]*end|20\d{2}))?""",
    "unit_hint": ["MEUR","EURm","€m","EUR million","mEUR","M€"]
  },
  "Total Scope 1+2 emissions (tCO2e / ktCO2e)": {
    "aliases": ["total scope 1 and 2 emissions","scope 1 & 2 emissions","scopes 1,2 emissions","operational emissions","ghg emissions (scope 1 and 2)","combined scope 1 and 2","location-based scope 1+2","market-based scope 1+2","gross scope 1 and scope 2 ghg emissions"],
    "regex": r"""\b(?:total\s+)?(?:scope|scopes)\s*1\s*(?:&|and|,)?\s*2(?:\s*(?:ghg|greenhouse\s+gas))?\s*emissions\b(?:[^\w%]*(?:location[-\s]*based|market[-\s]*based))?(?:[^\w%]*(?:tCO2e|ktCO2e|t\s*CO2e|tonne[s]?\s*CO2e|metric\s*ton[s]?\s*CO2e|mtCO2e|CO2e))?""",
    "unit_hint": ["tCO2e","ktCO2e","tonnes CO2e","metric tons CO2e","mtCO2e","CO2e"]
  },
  "Scope 3 emissions (tCO2e / ktCO2e)": {
    "aliases": ["scope 3 emissions","gross scope 3 ghg emissions","indirect emissions (scope 3)","s3 emissions","total scope 3","scope iii emissions"],
    "regex": r"""\b(?:total\s+)?(?:scope|scopes)\s*3(?:\s*(?:ghg|greenhouse\s+gas))?\s*emissions\b(?:[^\w%]*(?:tCO2e|ktCO2e|t\s*CO2e|tonne[s]?\s*CO2e|metric\s*ton[s]?\s*CO2e|mtCO2e|CO2e))?""",
    "unit_hint": ["tCO2e","ktCO2e","tonnes CO2e","metric tons CO2e","mtCO2e","CO2e"]
  },
  "Total salaries & remuneration expense (MEUR)": {
    "aliases": ["personnel expenses","employee benefit expenses","staff costs","wages and salaries","wages salaries and social costs","payroll costs","remuneration expenses","employee costs","salary expenses","total personnel cost","personnel expense"],
    "regex": r"""\b(?:personnel|employee|staff|payroll|remuneration)\s+(?:expense[s]?|cost[s]?|benefit[s]?)(?:[^\w]+(?:MEUR|EURm|€m|EUR\s*million|mEUR|M€))?""",
    "unit_hint": ["MEUR","EURm","€m","EUR million","mEUR","M€"]
  },
  "Audit fees (MEUR)": {
    "aliases": ["audit fees","fees to auditor","statutory audit fees","audit remuneration"],
    "regex": r"""\b(?:audit|auditor|statutory\s+audit)\s+fee[s]?\b(?:[^\w]+(?:MEUR|EURm|€m|EUR\s*million|mEUR|M€))?""",
    "unit_hint": ["MEUR","EURm","€m","EUR million","mEUR","M€"]
  },
  "Non-audit fees (MEUR)": {
    "aliases": ["non-audit fees","non audit fees","fees for non-audit services","other services by auditor","audit-related fees (non-audit)","nonassurance fees"],
    "regex": r"""\b(?:non[-\s]?audit|nonassurance|other\s+services)\s+fee[s]?\b(?:[^\w]+(?:MEUR|EURm|€m|EUR\s*million|mEUR|M€))?""",
    "unit_hint": ["MEUR","EURm","€m","EUR million","mEUR","M€"]
  },
  "Earnings per share, diluted (EUR)": {
    "aliases": ["earnings per share diluted","diluted earnings per share","diluted eps","eps (diluted)","eps diluted"],
    "regex": r"""\b(?:earnings\s+per\s+share|eps)\b(?:[^\w]+)?(?:diluted)\b(?:[^\w]+(?:EUR|€|EUR/share))?""",
    "unit_hint": ["EUR per share","EUR","€"]
  },
  "Board meetings per year (count)": {
    "aliases": ["board meetings per year","number of board meetings","total board meetings","meetings of the board","bod meetings"],
    "regex": r"""\b(?:board(?:\s+of\s+directors)?|bod)\s+meeting[s]?(?:\s*(?:per\s+year|in\s+the\s+year|during\s+the\s+year))?""",
    "unit_hint": ["count","number"]
  },
}

def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[\u2010-\u2015\u2212]", "-", s)
    s = re.sub(r"[^a-z0-9%€/.\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _compile_patterns(kpi_patterns: Dict[str, Dict]) -> Dict[str, Dict]:
    compiled = {}
    for kpi, spec in kpi_patterns.items():
        rx = re.compile(spec["regex"], re.IGNORECASE | re.VERBOSE)
        compiled[kpi] = {**spec, "regex": rx}
    return compiled

KPI_RX = _compile_patterns(KPI_PATTERNS)

def which_kpi(header_text: str) -> Optional[str]:
    h = _norm(header_text or "")
    for kpi, spec in KPI_RX.items():
        for alias in spec["aliases"]:
            if _norm(alias) in h:
                return kpi
    for kpi, spec in KPI_RX.items():
        if spec["regex"].search(header_text or ""):
            return kpi
    return None

def extract_kpis(df_wide: pd.DataFrame) -> pd.DataFrame:
    def _clean_item_text(s: str) -> str:
        t = (s or "").strip()
        t = re.sub(r"\s+", " ", t)
        # remove leading repeated section fragments like "for the ye..."
        t = re.sub(r"^(for the ye\w*|for the year.*?:?)\s*", "", t, flags=re.IGNORECASE)
        # trim obvious filler
        t = re.sub(r"^(this section|several m.*|of \d+\.?\d*%.*)", "", t, flags=re.IGNORECASE)
        return t.strip()

    def _plausible_for_kpi(kpi: str, item_txt: str) -> bool:
        it = _clean_item_text(item_txt).lower()
        # Disallow noisy lines
        if any(bad in it for bad in ["note", "eur million", "%", "tax on", "transfer", "actuarial", "inventory", "hedging", "challenge"]):
            return False
        if kpi == "Sales (MEUR)":
            # require the keyword to be early in the string
            return bool(re.search(r"^(net\s+)?(sales|revenue|turnover)\b", it))
        if kpi.startswith("Earnings per share"):
            return "earnings per share" in it or re.search(r"\beps\b", it)
        return True
    if "Item" not in df_wide.columns:
        return pd.DataFrame(columns=["KPI","Year","Value","Section","Item","Unit"])
    year_cols = [c for c in df_wide.columns if re.fullmatch(r"(19|20)\d{2}", str(c))]
    year_cols_sorted = sorted(year_cols, reverse=True)

    out = []
    for _, row in df_wide.iterrows():
        item_txt = str(row["Item"])
        kpi = which_kpi(item_txt)
        if not kpi:
            continue
        if not _plausible_for_kpi(kpi, item_txt):
            continue
        year_used, val = None, None
        for y in year_cols_sorted:
            v = row.get(y)
            if pd.notna(v):
                val = float(v); year_used = y; break
        if val is None:
            right_cols = [c for c in df_wide.columns if c not in ("Section","Item","Note")]
            for c in right_cols:
                v = row.get(c)
                if pd.notna(v):
                    val = float(v); year_used = c if c in year_cols_sorted else None; break
        unit = KPI_PATTERNS[kpi]["unit_hint"][0] if KPI_PATTERNS[kpi].get("unit_hint") else None
        out.append({
            "KPI": kpi, "Year": year_used, "Value": val,
            "Section": row.get("Section"), "Item": item_txt, "Unit": unit
        })

    if not out:
        return pd.DataFrame(columns=["KPI","Year","Value","Section","Item","Unit"])
    kpi_df = pd.DataFrame(out)
    # Prefer higher Value for duplicated KPI within a page (helps avoid small spurious matches)
    kpi_df = (kpi_df.sort_values(["KPI","Value","Year"], ascending=[True, False, False])
                    .groupby("KPI", as_index=False).first())
    return kpi_df

def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]+", "_", s)

def _px_bbox_to_pdf_points(bbox_px: Tuple[int, int, int, int], img_shape: Tuple[int, int], dpi: int = DPI) -> Tuple[float, float, float, float]:
    """Convert pixel bbox (x1,y1,x2,y2) from a top-left origin image to PDF points (72 dpi) with bottom-left origin.
    Returns (x1_pt, y1_pt, x2_pt, y2_pt) in PDF coordinate space.
    """
    x1, y1, x2, y2 = bbox_px
    height_px, width_px = img_shape[0], img_shape[1]
    px_to_pt = 72.0 / float(dpi)
    page_h_pt = height_px * px_to_pt
    x1_pt = x1 * px_to_pt
    x2_pt = x2 * px_to_pt
    # invert Y for bottom-left origin
    y1_pt = page_h_pt - (y2 * px_to_pt)
    y2_pt = page_h_pt - (y1 * px_to_pt)
    return (x1_pt, y1_pt, x2_pt, y2_pt)

def _areas_from_lp(image_path: Path) -> List[str]:
    """Run LP detection on the page image and return Camelot table_areas strings (x1,y1,x2,y2 in points)."""
    if lp_model is None:
        return []
    image = cv2.imread(str(image_path))
    if image is None:
        return []
    h, w = image.shape[:2]
    layout = lp_model.detect(image)
    table_blocks = [b for b in layout if b.type == "Table"]
    areas: List[str] = []
    for block in table_blocks:
        x1, y1, x2, y2 = map(int, block.coordinates)
        x1 = max(0, x1 - LP_PADDING); y1 = max(0, y1 - LP_PADDING)
        x2 = min(w, x2 + LP_PADDING); y2 = min(h, y2 + LP_PADDING)
        if (x2 - x1) < LP_MIN_WIDTH or (y2 - y1) < LP_MIN_HEIGHT:
            continue
        x1p, y1p, x2p, y2p = _px_bbox_to_pdf_points((x1, y1, x2, y2), (h, w), DPI)
        areas.append(f"{x1p:.2f},{y1p:.2f},{x2p:.2f},{y2p:.2f}")
    return areas

def _pad_area_pts(area: str, pad: float) -> str:
    x1, y1, x2, y2 = map(float, area.split(','))
    return f"{max(0.0, x1 - pad):.2f},{max(0.0, y1 - pad):.2f},{x2 + pad:.2f},{y2 + pad:.2f}"

def _score_dataframe_quality(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.0
    # Non-empty cell ratio
    non_empty_ratio = 1.0 - df.replace('', pd.NA).isna().mean().mean()
    # Numeric plausibility: fraction of cells parsable as numbers
    def _to_num(v):
        try:
            s = str(v)
            s = s.replace('%','')
            s = s.replace('\u202f',' ').replace('\xa0',' ').replace(',','.')
            s = re.sub(r"(?<=\d)[ ,](?=\d{3}\b)", "", s)
            return float(s)
        except Exception:
            return np.nan
    numeric_ratio = df.applymap(_to_num).notna().mean().mean()
    # Column consistency: prefer tables with at least 2 columns and <= 40 columns
    col_bonus = 0.0
    if 2 <= df.shape[1] <= 40:
        col_bonus = 0.1
    # Rows bonus
    row_bonus = 0.0
    if df.shape[0] >= 3:
        row_bonus = 0.05
    return float(non_empty_ratio)*0.55 + float(numeric_ratio)*0.35 + col_bonus + row_bonus

def _choose_best_camelot(pdf_path: Path, page_index_1based: int, table_areas: Optional[List[str]]) -> Optional[Dict]:
    has_gs = bool(shutil.which("gswin64c") or shutil.which("gswin32c") or shutil.which("gs"))
    flavors = ["lattice", "stream"] if has_gs else ["stream"]
    area_sets = [table_areas] if table_areas else [None]
    # Also sweep small paddings if we have areas
    if table_areas:
        paddings = [0.0, 2.0, 4.0, 6.0]
        padded_sets = []
        for pad in paddings:
            padded_sets.append([_pad_area_pts(a, pad) for a in table_areas])
        area_sets = padded_sets

    stream_grid = [
        {"row_tol": r, "column_tol": c}
        for r in (5, 10, 15)
        for c in (5, 10, 15)
    ]
    lattice_grid = [
        {"line_scale": ls}
        for ls in (30, 40, 50, 60)
    ]

    best = {"score": -1.0, "tables": None}
    for areas in area_sets:
        for flavor in flavors:
            grid = lattice_grid if flavor == "lattice" else stream_grid
            for params in grid:
                try:
                    tables = camelot.read_pdf(
                        str(pdf_path), pages=str(page_index_1based), flavor=flavor,
                        table_areas=areas, strip_text="\n", **params
                    )
                except Exception:
                    continue
                if not tables or len(tables) == 0:
                    continue
                # Score by summing table quality
                total_score = 0.0
                for t in tables:
                    df_raw = getattr(t, 'df', None)
                    if isinstance(df_raw, pd.DataFrame):
                        total_score += _score_dataframe_quality(df_raw)
                if total_score > best["score"]:
                    best = {"score": total_score, "tables": tables, "flavor": flavor, "params": params, "areas": areas}
    return best if best["tables"] is not None else None

def extract_from_pdf_page(pdf_path: Path, page_index_1based: int, image_path: Path, out_root: Path):
    """Use LayoutParser to find table areas and Camelot to extract tables from the PDF page.
    Saves per-table CSV and returns (tables_found, kpi_df).
    """
    out_tables = out_root / "tables"
    out_csv = out_root / "csv"
    out_logs = out_root / "logs"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_csv.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    table_areas = _areas_from_lp(image_path)
    tables_found = 0
    kpi_results = []

    best = _choose_best_camelot(pdf_path, page_index_1based, table_areas)
    flavor_tag = (best.get("flavor") if best else None) or "best"
    if best and best.get("tables"):
        tables = best["tables"]
    else:
        tables = []

    for t_idx, t in enumerate(tables, start=1):
            try:
                df_raw = t.df
                if df_raw is None or df_raw.empty:
                    continue
                # Clean and choose best interpretation using existing helpers
                df_wide, df_long = build_wide_and_long(df_raw)
            except Exception as e:
                (out_logs / f"page_{page_index_1based:03d}_t{t_idx:02d}_{flavor_tag}_parse_error.txt").write_text(
                    f"{e}\n\n{traceback.format_exc()}", encoding="utf-8"
                )
                continue

            tables_found += 1
            base_name = f"page_{page_index_1based:03d}__{flavor_tag}__t{tables_found:02d}"
            df_wide.to_csv(out_csv / f"{base_name}__wide.csv", index=False, encoding="utf-8")
            df_long.to_csv(out_csv / f"{base_name}__long.csv", index=False, encoding="utf-8")

            kpi_df = extract_kpis(df_wide)
            if len(kpi_df):
                kpi_df.insert(0, "SourceImage", image_path.name)
                kpi_df.insert(1, "TableIndex", tables_found)
                kpi_results.append(kpi_df)

    kpis = pd.concat(kpi_results, ignore_index=True) if kpi_results else pd.DataFrame()
    return tables_found, kpis

def _process_one_page(args: Tuple[Path, int, Path, Path]) -> Tuple[int, int, Optional[pd.DataFrame], Optional[str]]:
    pdf_path, idx1, imgp, target_root = args
    try:
        tables_found, kpis = extract_from_pdf_page(pdf_path, idx1, imgp, target_root)
        return (idx1, tables_found, kpis, None)
    except Exception as e:
        return (idx1, 0, None, f"{e}")

def process_pdf(pdf_path: Path):
    pdf_name = pdf_path.stem
    print(f"\n=== Processing PDF: {pdf_name} ===")

    # 1) PDF -> pages
    pages_dir = PAGES_DIR / pdf_name
    page_imgs_all = pdf_to_pages(pdf_path, pages_dir)
    # Apply optional page range
    if PAGE_START is not None or PAGE_END is not None:
        start = max(1, PAGE_START or 1)
        end = PAGE_END or len(page_imgs_all)
        page_imgs = [p for i, p in enumerate(page_imgs_all, 1) if start <= i <= end]
    else:
        page_imgs = page_imgs_all
    print(f"  Converted {len(page_imgs)} pages -> {pages_dir}")

    # 2) Camelot extraction per page using LP-detected areas
    target_root = EXTRACT_ROOT / pdf_name
    target_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    all_kpis = []
    # Parallel processing with 3 workers
    tasks = [(pdf_path, i, p, target_root) for i, p in enumerate(page_imgs, 1)]
    # Collect results in order
    results: Dict[int, Tuple[int, Optional[pd.DataFrame], Optional[str]]] = {}
    with ProcessPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(_process_one_page, t): t for t in tasks}
        for fut in as_completed(futures):
            idx1, tables_found, kpis, err = fut.result()
            results[idx1] = (tables_found, kpis, err)
    for i in range(1, len(page_imgs)+1):
        tables_found, kpis, err = results.get(i, (0, None, "Missing"))
        imgp = page_imgs[i-1]
        if err:
            print(f"  [{i}/{len(page_imgs)}] {imgp.name}: ERROR {err}")
            (target_root / "logs").mkdir(exist_ok=True, parents=True)
            (target_root / "logs" / f"{safe_name(imgp.stem)}__fatal.txt").write_text(
                f"{err}", encoding="utf-8"
            )
        else:
            manifest.append({"page": i, "image": imgp.name, "tables_found": int(tables_found)})
            if kpis is not None and len(kpis):
                all_kpis.append(kpis)
            print(f"  [{i}/{len(page_imgs)}] {imgp.name}: tables={tables_found}")

    # Save manifest
    (target_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Save combined KPIs
    if all_kpis:
        kdf = pd.concat(all_kpis, ignore_index=True)
        kdf.to_csv(target_root / "combined_kpis.csv", index=False, encoding="utf-8")
        piv = (kdf.pivot_table(index=["SourceImage","TableIndex"], columns="KPI",
                               values="Value", aggfunc="first").reset_index())
        piv.to_csv(target_root / "combined_kpis_pivot.csv", index=False, encoding="utf-8")
    else:
        (target_root / "NO_KPIs_FOUND.txt").write_text(
            "No KPI matches found. Extend KPI_PATTERNS aliases/regex as needed.", encoding="utf-8"
        )

    # Summary
    print("\n--- SUMMARY ---")
    print(f"PDF: {pdf_path.name}")
    print(f"Pages -> {pages_dir}")
    print(f"Crops -> (not used; Camelot areas from LP directly)")
    print(f"Extracted -> {target_root}")
    print(f"  - manifest.json")
    if all_kpis:
        print(f"  - combined_kpis.csv")
        print(f"  - combined_kpis_pivot.csv")
    print(f"  - tables/ (PNGs + HTML)")
    print(f"  - csv/ (per-table wide/long)")
    print(f"  - logs/ (if any)")

def main():
    pdfs = sorted([p for p in REPORTS_DIR.glob("*.pdf")])
    if not pdfs:
        print(f"No PDFs found in: {REPORTS_DIR}")
        print("Put your reports (PDFs) in that folder and run again.")
        return
    for pdf in pdfs:
        process_pdf(pdf)
    print("\nAll done.")

if __name__ == "__main__":
    main()