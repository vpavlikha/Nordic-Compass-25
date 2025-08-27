# -*- coding: utf-8 -*-
"""
Robust table extractor (PP-Structure -> HTML -> pandas) + KPI labeling.
- Handles grayscale/alpha images
- Builds multi-row headers, flattens them (respects rowspan/colspan)
- Detects section rows, hyphen placeholders, thousands separators, percents
- Detects year columns and returns both wide and tidy-long formats
- Maps row labels to canonical KPIs using alias lists + robust regex
"""

from pathlib import Path
import re
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import cv2
from bs4 import BeautifulSoup
from paddleocr import PPStructure

# ----------- CONFIG -----------
IMG_PATH = r"C:\Users\vikto\Desktop\Nordic Compass\Screenshots\Accounting Statements\EvliP.png"
SHOW_LOG = False
LANG = "en"

# ----------- PP-Structure engine -----------
table_engine = PPStructure(
    show_log=SHOW_LOG,
    ocr_version="PP-OCRv4",
    image_orientation=False,
    layout=True,     # keep True so OCR inside tables is enabled
    table=True,
    lang=LANG,
)

# ----------- helpers -----------

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

def extract_table_html_blocks(img_path: str) -> List[str]:
    img = load_bgr_3ch(img_path)
    blocks = table_engine(img, return_ocr_result_in_table=True)
    html_list = []
    for b in blocks:
        if b.get("type") == "table" and isinstance(b.get("res"), dict):
            h = b["res"].get("html")
            if h:
                html_list.append(h)
    if not html_list:
        raise RuntimeError("No tables detected by PP-Structure.")
    return html_list

def _clean_html_for_read_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # normalize soft hyphen & non-breaking spaces
    for t in soup.find_all(text=True):
        if t.string:
            t.replace_with(
                t.string.replace("\u00ad", " ").replace("\xa0", " ").replace("\u202f", " ")
            )
    return str(soup)

def _try_multiheader_read(html: str) -> List[pd.DataFrame]:
    """
    Try to read with different header depths: [0], [0,1], [0,1,2].
    Keep only reasonably sized tables.
    """
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
    """
    Flatten simple or MultiIndex columns by joining non-empty parts.
    Also forward-fills blanks across levels (common with merged header cells).
    """
    if not isinstance(cols, pd.MultiIndex):
        return [str(c).strip() for c in cols]

    tuples = [list(map(lambda x: "" if (x is None or str(x).lower()=="nan") else str(x).strip(), tup))
              for tup in cols.to_list()]

    # propagate higher-level labels downward if lower level is blank
    for j in range(1, len(cols.levels)):
        for i, tup in enumerate(tuples):
            if tup[j] == "":
                tup[j] = tup[j-1]

    flat = [" | ".join([p for p in tup if p]) for tup in tuples]
    return [re.sub(r"\s{2,}", " ", f).strip(" |") for f in flat]

def _is_section_row(row: pd.Series, numeric_cols: List[str]) -> bool:
    """
    A 'section' row is where all numeric columns are empty and the first column has text.
    """
    first = str(row.iloc[0]).strip()
    if not first or first.lower() in ("nan",):
        return False
    if not numeric_cols:
        return False
    all_blank = True
    for c in numeric_cols:
        val = str(row.get(c, "")).strip()
        if val not in ("", "–", "-", "—", "nan"):
            all_blank = False
            break
    return all_blank

def _parse_number(x: str) -> Optional[float]:
    if x is None:
        return np.nan
    s = str(x).strip()
    if s in ("", "-", "–", "—"):
        return np.nan
    # keep percents as numeric fraction (optional). Here we’ll just strip % and keep numeric
    s = s.replace("%", "")
    # remove thousands separators (comma or space/thin space)
    s = s.replace("\u202f", " ").replace("\xa0", " ")
    s = re.sub(r"(?<=\d)[ ,](?=\d{3}\b)", "", s)  # 96,084 -> 96084 ; 3 283 -> 3283
    # change comma decimal to dot if present
    s = s.replace(",", ".")
    # parentheses: (123) -> -123
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
    """
    Pick the 'best' table by number of cells, then flatten headers and clean.
    """
    cands = []
    for h in html_list:
        cands.extend(_try_multiheader_read(h))
    if not cands:
        raise RuntimeError("Could not parse any table HTML.")

    best = max(cands, key=lambda d: d.shape[0] * d.shape[1])

    # Flatten columns
    best.columns = _flatten_columns(best.columns)

    # Drop fully empty rows
    best = best[(best.astype(str).replace("", np.nan)).notna().any(axis=1)].reset_index(drop=True)

    # Strip whitespace in string cells
    best = best.applymap(lambda v: v.strip() if isinstance(v, str) else v)

    return best

def build_wide_and_long(df_in: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Heuristics:
      - First non-numeric-ish column is 'Item'
      - Optional 'Note' column looks like 1, 2.3, 2.14
      - Detect year columns from header names; the rest treated as numeric data columns
      - Build 'Section' by detecting section-only rows and ffill
    """
    df = df_in.copy()

    # detect numeric-ish columns (by values)
    numeric_likelihood = {}
    for c in df.columns:
        s = df[c].astype(str).head(30)
        nums = s.map(lambda x: _parse_number(x)).notna().mean()
        numeric_likelihood[c] = nums
    # Item is the lowest numeric likelihood
    item_col = min(numeric_likelihood, key=numeric_likelihood.get)
    if "Item" not in df.columns:
        df.rename(columns={item_col: "Item"}, inplace=True)

    # Note-like column
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

    # Year columns by header name
    year_cols = []
    year_name_map = {}
    for c in df.columns:
        if c in ("Item", "Note"):
            continue
        y = _looks_year_name(c)
        if y:
            year_cols.append(c)
            year_name_map[c] = y

    # If still none, treat all remaining non-Item/Note as numeric
    if not year_cols:
        year_cols = [c for c in df.columns if c not in ("Item", "Note")]

    # Parse numerics for year/measure columns
    for c in year_cols:
        df[c] = df[c].map(_parse_number)

    # Detect section rows
    numeric_cols_for_section = year_cols.copy()
    section = []
    cur = None
    for _, row in df.iterrows():
        if _is_section_row(row, numeric_cols_for_section):
            cur = str(row["Item"]).strip()
            section.append(cur)
        else:
            section.append(cur)
    df.insert(0, "Section", section)

    # Drop pure section rows from the numeric table
    mask_section_only = df.apply(lambda r: _is_section_row(r, numeric_cols_for_section), axis=1)
    df_wide = df.loc[~mask_section_only].reset_index(drop=True)

    # Normalize year names & ordering
    df_wide = df_wide.rename(columns=year_name_map)
    years_in_cols = sorted([c for c in df_wide.columns if re.fullmatch(r"(19|20)\d{2}", str(c))], reverse=True)
    lead = ["Section", "Item"] + (["Note"] if "Note" in df_wide.columns else [])
    others = [c for c in df_wide.columns if c not in lead + years_in_cols]
    df_wide = df_wide[lead + years_in_cols + others]

    # Tidy/long
    if years_in_cols:
        df_long = df_wide.melt(
            id_vars=lead,
            value_vars=years_in_cols,
            var_name="Year",
            value_name="Value",
        ).dropna(subset=["Value"]).reset_index(drop=True)
    else:
        numeric_cols = [c for c in df_wide.columns if c not in lead]
        df_long = df_wide.melt(
            id_vars=lead,
            value_vars=numeric_cols,
            var_name="Measure",
            value_name="Value",
        ).dropna(subset=["Value"]).reset_index(drop=True)

    # Clean item text a bit
    df_wide["Item"] = (
        df_wide["Item"].astype(str)
        .str.replace(r"\.{2,}", "", regex=True)
        .str.replace(r"\s{2,}", " ", regex=True)
        .str.strip()
    )
    df_long["Item"] = (
        df_long["Item"].astype(str)
        .str.replace(r"\.{2,}", "", regex=True)
        .str.replace(r"\s{2,}", " ", regex=True)
        .str.strip()
    )

    return df_wide, df_long

# ----------- KPI patterns + matchers -----------

KPI_PATTERNS: Dict[str, Dict] = {
  # 1) Sales / Revenue (MEUR)
  "Sales (MEUR)": {
    "aliases": [
      "sales", "net sales", "revenue", "revenues", "net revenue",
      "turnover", "net turnover", "total revenue", "group revenue",
      "sales revenue", "operating revenue", "income from sales"
    ],
    "regex": r"""
      \b(?:net\s+)?(?:sales|revenue|turnover|group\s+revenue|total\s+revenue)\b
      (?:[^A-Za-z0-9]+(?:MEUR|EURm|€m|EUR\s*million|mEUR|M€))?
      (?:[^A-Za-z0-9]+(?:FY|YE|year[-\s]*end|20\d{2}))?
    """,
    "unit_hint": ["MEUR","EURm","€m","EUR million","mEUR","M€"]
  },

  # 2) Total Scope 1 & 2 GHG emissions (combined)
  "Total Scope 1+2 emissions (tCO2e / ktCO2e)": {
    "aliases": [
      "total scope 1 and 2 emissions", "scope 1 & 2 emissions",
      "scopes 1,2 emissions", "operational emissions",
      "ghg emissions (scope 1 and 2)", "combined scope 1 and 2",
      "location-based scope 1+2", "market-based scope 1+2",
      "gross scope 1 and scope 2 ghg emissions"
    ],
    "regex": r"""
      \b(?:total\s+)?(?:scope|scopes)\s*1\s*(?:&|and|,)?\s*2
      (?:\s*(?:ghg|greenhouse\s+gas))?\s*emissions\b
      (?:[^\w%]*(?:location[-\s]*based|market[-\s]*based))?
      (?:[^\w%]*(?:tCO2e|ktCO2e|t\s*CO2e|tonne[s]?\s*CO2e|metric\s*ton[s]?\s*CO2e|mtCO2e|CO2e))?
    """,
    "unit_hint": ["tCO2e","ktCO2e","tonnes CO2e","metric tons CO2e","mtCO2e","CO2e"]
  },

  # 3) Scope 3 GHG emissions (total)
  "Scope 3 emissions (tCO2e / ktCO2e)": {
    "aliases": [
      "scope 3 emissions", "gross scope 3 ghg emissions",
      "indirect emissions (scope 3)", "s3 emissions", "total scope 3",
      "scope iii emissions"
    ],
    "regex": r"""
      \b(?:total\s+)?(?:scope|scopes)\s*3
      (?:\s*(?:ghg|greenhouse\s+gas))?\s*emissions\b
      (?:[^\w%]*(?:tCO2e|ktCO2e|t\s*CO2e|tonne[s]?\s*CO2e|metric\s*ton[s]?\s*CO2e|mtCO2e|CO2e))?
    """,
    "unit_hint": ["tCO2e","ktCO2e","tonnes CO2e","metric tons CO2e","mtCO2e","CO2e"]
  },

  # 4) Total salaries & remuneration expense (MEUR)
  "Total salaries & remuneration expense (MEUR)": {
    "aliases": [
      "personnel expenses", "employee benefit expenses", "staff costs",
      "wages and salaries", "wages salaries and social costs",
      "payroll costs", "remuneration expenses", "employee costs",
      "salary expenses", "total personnel cost", "personnel expense"
    ],
    "regex": r"""
      \b(?:personnel|employee|staff|payroll|remuneration)\s+
      (?:expense[s]?|cost[s]?|benefit[s]?)
      (?:[^\w]+(?:MEUR|EURm|€m|EUR\s*million|mEUR|M€))?
    """,
    "unit_hint": ["MEUR","EURm","€m","EUR million","mEUR","M€"]
  },

  # 5) Audit fees (MEUR)
  "Audit fees (MEUR)": {
    "aliases": [
      "audit fees", "fees to auditor", "statutory audit fees",
      "audit remuneration"
    ],
    "regex": r"""
      \b(?:audit|auditor|statutory\s+audit)\s+fee[s]?\b
      (?:[^\w]+(?:MEUR|EURm|€m|EUR\s*million|mEUR|M€))?
    """,
    "unit_hint": ["MEUR","EURm","€m","EUR million","mEUR","M€"]
  },

  # 6) Non-audit fees (MEUR)
  "Non-audit fees (MEUR)": {
    "aliases": [
      "non-audit fees", "non audit fees", "fees for non-audit services",
      "other services by auditor", "audit-related fees (non-audit)",
      "nonassurance fees"
    ],
    "regex": r"""
      \b(?:non[-\s]?audit|nonassurance|other\s+services)\s+fee[s]?\b
      (?:[^\w]+(?:MEUR|EURm|€m|EUR\s*million|mEUR|M€))?
    """,
    "unit_hint": ["MEUR","EURm","€m","EUR million","mEUR","M€"]
  },

  # 7) EPS diluted (EUR)
  "Earnings per share, diluted (EUR)": {
    "aliases": [
      "earnings per share diluted", "diluted earnings per share",
      "diluted eps", "eps (diluted)", "eps diluted"
    ],
    "regex": r"""
      \b(?:earnings\s+per\s+share|eps)\b
      (?:[^\w]+)?(?:diluted)\b
      (?:[^\w]+(?:EUR|€|EUR/share))?
    """,
    "unit_hint": ["EUR per share","EUR","€"]
  },

  # 8) Board meetings per year (count)
  "Board meetings per year (count)": {
    "aliases": [
      "board meetings per year", "number of board meetings",
      "total board meetings", "meetings of the board",
      "board of directors meetings", "bod meetings"
    ],
    "regex": r"""
      \b(?:board(?:\s+of\s+directors)?|bod)\s+meeting[s]?
      (?:\s*(?:per\s+year|in\s+the\s+year|during\s+the\s+year))?
    """,
    "unit_hint": ["count","number"]
  },
}

def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[\u2010-\u2015\u2212]", "-", s)   # unify dashes
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
    """Return canonical KPI if header_text matches."""
    h = _norm(header_text or "")
    # quick alias pass
    for kpi, spec in KPI_RX.items():
        for alias in spec["aliases"]:
            if _norm(alias) in h:
                return kpi
    # regex fallback
    for kpi, spec in KPI_RX.items():
        if spec["regex"].search(header_text or ""):
            return kpi
    return None

def parse_value_and_unit(text: str, unit_hint: List[str] | None = None):
    """If you have text with embedded value/unit. Here we mostly use numeric cells already."""
    s = text.strip()
    s_num = (s.replace("\xa0"," ").replace("\u202f"," ").replace(" ", ""))
    s_num = re.sub(r"[^0-9,\.\-\(\)]", "", s_num)
    if s_num.count(",") == 1 and s_num.count(".") == 0:
        s_num = s_num.replace(",", ".")
    if s_num.count(",") > 1 and s_num.count(".") == 0:
        s_num = s_num.replace(",", "")
    try:
        val = float(s_num.strip("()"))
        if s_num.startswith("(") and s_num.endswith(")"):
            val = -val
    except:
        val = None
    unit = None
    if unit_hint:
        unit = next((u for u in unit_hint if u.lower() in s.lower()), None)
    return val, unit

def extract_kpis(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Try to label rows as KPIs by their 'Item' text and take the latest-year value.
    Returns a DataFrame: KPI | Year | Value | Section | Item | Unit
    """
    if "Item" not in df_wide.columns:
        return pd.DataFrame(columns=["KPI","Year","Value","Section","Item","Unit"])

    # figure out year ordering
    year_cols = [c for c in df_wide.columns if re.fullmatch(r"(19|20)\d{2}", str(c))]
    year_cols_sorted = sorted(year_cols, reverse=True)

    out = []
    for _, row in df_wide.iterrows():
        item_txt = str(row["Item"])
        kpi = which_kpi(item_txt)
        if not kpi:
            continue

        # pick latest available year value
        year_used = None
        val = None
        for y in year_cols_sorted:
            v = row.get(y)
            if pd.notna(v):
                val = float(v)
                year_used = y
                break

        if val is None:
            # fallback: first numeric column to the right of Item/Note
            right_cols = [c for c in df_wide.columns if c not in ("Section","Item","Note")]
            for c in right_cols:
                v = row.get(c)
                if pd.notna(v):
                    val = float(v)
                    year_used = c if c in year_cols_sorted else None
                    break

        unit = KPI_PATTERNS[kpi]["unit_hint"][0] if KPI_PATTERNS[kpi].get("unit_hint") else None
        out.append({
            "KPI": kpi,
            "Year": year_used,
            "Value": val,
            "Section": row.get("Section"),
            "Item": item_txt,
            "Unit": unit
        })

    # deduplicate: keep the first (most recent) per KPI
    if not out:
        return pd.DataFrame(columns=["KPI","Year","Value","Section","Item","Unit"])
    kpi_df = pd.DataFrame(out)
    kpi_df = (kpi_df.sort_values(["KPI", "Year"], ascending=[True, False])
                    .groupby("KPI", as_index=False)
                    .first())
    return kpi_df

# ----------- run -----------
if __name__ == "__main__":
    p = Path(IMG_PATH)
    if not p.exists():
        raise FileNotFoundError(p)

    html_list = extract_table_html_blocks(str(p))
    df_best = html_to_best_df(html_list)
    df_wide, df_long = build_wide_and_long(df_best)

    print("\n=== WIDE (flattened header) ===")
    with pd.option_context("display.max_rows", 80, "display.max_columns", None, "display.width", 240):
        print(df_wide.head(30).to_string(index=False))

    print("\n=== LONG/TIDY (Section, Item, [Note], Year, Value) ===")
    with pd.option_context("display.max_rows", 80, "display.max_columns", None, "display.width", 240):
        print(df_long.head(40).to_string(index=False))

    # --- KPI extraction ---
    kpis = extract_kpis(df_wide)
    print("\n=== KPIs (auto-labeled; latest year if present) ===")
    if len(kpis):
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
            print(kpis.to_string(index=False))
    else:
        print("No KPI matches found in this table. Try extending KPI_PATTERNS['aliases'] for your dataset.")
