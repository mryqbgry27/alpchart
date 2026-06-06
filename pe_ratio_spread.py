"""
==============================================================================
  Relative P/E Ratio Comparison
  --------------------------------
  Compares two assets by computing daily Trailing Twelve-Month (TTM) P/E
  ratios and charting their relative relationship over time.

  Metric  :  PE_Ratio = PE_A / PE_B
             • Ratio > 1  →  Ticker A trades at a premium to Ticker B
             • Ratio < 1  →  Ticker A trades at a discount to Ticker B
             • Ratio = 1  →  Parity

  Note on currency: P/E = Price / EPS (both in the stock's native currency),
  so it is dimensionless and CHF conversion has no effect on the ratio.
  The currency note is retained in the chart for transparency.

  TTM EPS: sum of the most recent 4 reported quarterly Basic EPS values
  (falls back to Diluted EPS, then Net Income / Shares if Basic is absent).

  Usage:
      python pe_ratio_spread.py

  Dependencies:
      pip install yfinance plotly numpy scipy pandas
==============================================================================
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  —  edit these values, then just run: python pe_ratio_spread.py
# ─────────────────────────────────────────────────────────────────────────────

TICKER_A   = "AAPL"          # <- left-hand ticker  (numerator of the ratio)
TICKER_B   = "MSFT"          # <- right-hand ticker (denominator of the ratio)
START_DATE = "2012-01-01"    # <- YYYY-MM-DD  (longer history → more reliable SMAs)
END_DATE   = "today"         # <- YYYY-MM-DD  or  "today"

# ── Currency note ─────────────────────────────────────────────────────────────
# P/E ratios are currency-neutral (Price ÷ EPS, both in native currency).
# Setting this to True adds a [CHF] label to the chart title for transparency
# but does NOT change any calculation.
CONVERT_TO_CHF = True   # <- True | False

# ── Moving average windows applied to the PE ratio ───────────────────────────
# Each entry: (window_days, label, hex_colour)
SMA_WINDOWS = [
    ( 50, "50-day SMA  (Fast / Tactical)",   "#FFD700"),  # gold
    (200, "200-day SMA (Medium-Term Trend)",  "#00BFFF"),  # sky blue
    (600, "600-day SMA (Slow / Structural)",  "#FF69B4"),  # pink
]

# ── Reference levels on the PE ratio panel ───────────────────────────────────
# Each entry: (y_value, label, colour, plotly_dash_style)
# "A" and "B" in labels are replaced with the actual ticker names at runtime.
REFERENCE_LEVELS = [
    (1.00, "Parity  (A = B)",        "#888888", "solid"),
    (1.50, "A at +50% Premium",      "#FFA500", "dash"),
    (0.67, "A at −33% Discount",     "#FFA500", "dash"),
    (2.00, "A at +100% Premium",     "#FF4444", "dash"),
    (0.50, "A at −50% Discount",     "#FF4444", "dash"),
]

# ── Y-axis limits ─────────────────────────────────────────────────────────────
# Raw P/E panel (top): hard limits; None = automatic data-driven range with buffer
PE_Y_MIN     = None    # e.g.  5  |  None = automatic
PE_Y_MAX     = None    # e.g. 80  |  None = automatic

# Ratio panel (bottom): clip extreme ratio swings from early history
RATIO_Y_MIN  = None    # e.g. 0.3  |  None = automatic
RATIO_Y_MAX  = None    # e.g. 3.5  |  None = automatic

# ── EPS history ──────────────────────────────────────────────────────────────
# Maximum number of quarterly earnings periods to request from Yahoo Finance.
# get_earnings_dates(limit=N) is the primary source; higher values fetch more
# history (each unit ≈ 1 quarter).  80 covers ~20 years; 160 covers ~40 years.
EPS_HISTORY_LIMIT = 80   # <- increase if you need pre-2005 data

# ── Manual EPS override (optional) ──────────────────────────────────────────
# When Yahoo Finance history is short (~4-5 yrs), point this at a CSV file
# containing full historical quarterly EPS and it will be used instead.
# CSV must have two columns:  date (YYYY-MM-DD)  and  eps (quarterly EPS $).
# Example row:   2015-06-27, 1.85
# Free sources:  macrotrends.net  |  financialmodelingprep.com  |  simfin.com
EPS_CSV_PATH = None   # <- e.g. "aapl_eps.csv"  |  None = use Yahoo Finance

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("pe_charts")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def resolve_end_date(end: str) -> str:
    return datetime.today().strftime("%Y-%m-%d") if end.strip().lower() == "today" else end


def fetch_prices(ticker: str, start: str, end: str) -> "pd.Series | None":
    """Download Adjusted Close prices; return a DatetimeIndex Series or None."""
    print(f"  ↓ {ticker} prices …", end=" ", flush=True)
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    except Exception as exc:
        print(f"FAILED ({exc})"); return None
    if df is None or df.empty:
        print("FAILED (no data)"); return None
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)
    col = "Close" if "Close" in df.columns else df.columns[0]
    s   = df[col].dropna().astype(float)
    print(f"OK  ({len(s)} trading days)")
    return s


def _tz_strip(idx: pd.Index) -> pd.DatetimeIndex:
    """Remove timezone info from a DatetimeIndex for consistent comparisons."""
    dti = pd.DatetimeIndex(idx)
    return dti.tz_localize(None) if dti.tz is not None else dti


def get_quarterly_eps(ticker: str) -> "pd.Series | None":
    """
    Fetch quarterly EPS for TTM P/E calculation.

    Source priority
    ---------------
    0. EPS_CSV_PATH         ← user-supplied CSV; unlimited history
    1. get_earnings_dates   ← best yfinance source; up to ~20 yrs if Yahoo has it
    2. quarterly_income_stmt ← recent ~5 quarters only
    3. quarterly_earnings   ← legacy attribute
    4. income_stmt (annual) ← 4-5 fiscal years; each year split into 4 synthetic
                               quarters (annual EPS ÷ 4) so TTM math still works
    5. Net Income ÷ Shares  ← derived from balance sheet + income statement
    """
    t = yf.Ticker(ticker)

    def _clean(s: pd.Series, source: str) -> "pd.Series | None":
        try:
            s.index = _tz_strip(pd.DatetimeIndex(s.index))
            s = (s.dropna()
                  .astype(float)
                  .sort_index()
                  [lambda x: x.index <= pd.Timestamp.today()])
            if len(s) >= 2:
                print(f"    EPS [{source}]: {len(s)} quarters "
                      f"({s.index[0].date()} → {s.index[-1].date()})")
                return s
        except Exception:
            pass
        return None

    # ── Method 0: user-supplied CSV (unlimited history) ──────────────────────
    if EPS_CSV_PATH:
        try:
            csv_df = pd.read_csv(EPS_CSV_PATH, parse_dates=["date"], index_col="date")
            s = _clean(csv_df.iloc[:, 0], f"CSV / {EPS_CSV_PATH}")
            if s is not None:
                return s
            print(f"    ⚠️  CSV loaded but no valid EPS values found in '{EPS_CSV_PATH}'.")
        except Exception as exc:
            print(f"    ⚠️  Could not load EPS CSV '{EPS_CSV_PATH}': {exc}")

    # ── Method 1: get_earnings_dates — can cover many years if available ─────
    try:
        ed = t.get_earnings_dates(limit=EPS_HISTORY_LIMIT)
        if ed is not None and not ed.empty:
            print(f"    [diag] get_earnings_dates cols: {ed.columns.tolist()}  "
                  f"rows: {len(ed)}")
            for col in ("Reported EPS", "EPS Actual", "Actual EPS", "epsActual"):
                if col in ed.columns:
                    s = _clean(ed[col], f"get_earnings_dates / {col}")
                    if s is not None:
                        return s
        else:
            print(f"    [diag] get_earnings_dates returned empty")
    except Exception as exc:
        print(f"    [diag] get_earnings_dates failed: {exc}")

    # ── Method 2: quarterly income statement (~5 recent quarters) ────────────
    # Stash result; if annual data (Method 4) is longer we prefer that,
    # but we always fall back to this if nothing better is found.
    _quarterly_fallback = None
    for attr in ("quarterly_income_stmt", "quarterly_financials"):
        try:
            stmt = getattr(t, attr)
            if stmt is None or (hasattr(stmt, "empty") and stmt.empty):
                continue
            for label in ("Basic EPS", "Diluted EPS"):
                if label in stmt.index:
                    s = _clean(stmt.loc[label], f"{label} / {attr}")
                    if s is not None:
                        if len(s) >= 12:
                            return s   # plenty of history — use immediately
                        # Short history: stash and try annual for more coverage
                        if _quarterly_fallback is None:
                            print(f"    ⚠️  Only {len(s)} quarters from quarterly "
                                  f"statements — trying annual income_stmt …")
                            _quarterly_fallback = s
        except Exception:
            pass

    # ── Method 3: legacy quarterly_earnings attribute ─────────────────────────
    try:
        ea = t.quarterly_earnings
        if ea is not None and not ea.empty and "EPS" in ea.columns:
            s = _clean(ea["EPS"], "quarterly_earnings")
            if s is not None:
                return s
    except Exception:
        pass

    # ── Method 4: Annual income_stmt → synthetic quarterly EPS ───────────────
    # income_stmt (annual) typically covers 4-5 fiscal years.
    # We split each year's EPS into 4 equal synthetic quarters (annual ÷ 4)
    # so that TTM math (sum of last 4 quarters) still returns the correct
    # annual EPS at any date within that fiscal year.
    for attr in ("income_stmt", "financials"):
        try:
            stmt = getattr(t, attr)
            if stmt is None or (hasattr(stmt, "empty") and stmt.empty):
                continue
            for label in ("Basic EPS", "Diluted EPS"):
                if label in stmt.index:
                    annual = stmt.loc[label].dropna().astype(float)
                    annual.index = _tz_strip(pd.DatetimeIndex(annual.index))
                    annual = annual.sort_index()
                    annual = annual[annual.index <= pd.Timestamp.today()]
                    if len(annual) < 1:
                        continue

                    # Build synthetic quarterly series: 4 equal quarters per year
                    records: dict = {}
                    for fy_end, fy_eps in annual.items():
                        q_eps = fy_eps / 4.0
                        for months_back in (0, 3, 6, 9):
                            q_date = fy_end - pd.DateOffset(months=months_back)
                            records[q_date] = q_eps

                    q_series = (pd.Series(records)
                                  .sort_index()
                                  .dropna()
                                  [lambda x: x.index <= pd.Timestamp.today()])

                    if len(q_series) >= 4:
                        print(f"    EPS [{label} annual÷4 / {attr}]: "
                              f"{len(annual)} fiscal years → {len(q_series)} synthetic "
                              f"quarters ({q_series.index[0].date()} → "
                              f"{q_series.index[-1].date()})")
                        print(f"    ℹ️  P/E updates annually (set EPS_CSV_PATH for "
                              f"quarterly granularity going further back).")
                        return q_series
        except Exception:
            pass

    # Annual fallback also failed — use stashed short quarterly series if any
    if _quarterly_fallback is not None:
        print(f"    ℹ️  Using {len(_quarterly_fallback)}-quarter series "
              f"(annual data unavailable for this ticker).")
        return _quarterly_fallback

    # ── Method 5: Net Income ÷ Diluted Shares ────────────────────────────────
    try:
        stmt = (getattr(t, "quarterly_income_stmt", None)
                or getattr(t, "quarterly_financials", None))
        bs   = getattr(t, "quarterly_balance_sheet", None)
        if stmt is not None and bs is not None:
            ni_row = next((r for r in ("Net Income", "NetIncome")
                           if r in stmt.index), None)
            sh_row = next((r for r in ("Diluted Average Shares",
                                       "Ordinary Shares Number", "Share Issued")
                           if r in (stmt.index if stmt is not None else [])
                           or r in (bs.index   if bs   is not None else [])), None)
            if ni_row and sh_row:
                ni = stmt.loc[ni_row] if ni_row in stmt.index else bs.loc[ni_row]
                sh = stmt.loc[sh_row] if sh_row in stmt.index else bs.loc[sh_row]
                common = ni.index.intersection(sh.index)
                if len(common) >= 2:
                    s = _clean((ni.loc[common] / sh.loc[common]).rename("EPS"),
                               "Net Income / Shares")
                    if s is not None:
                        return s
    except Exception:
        pass

    print(f"    ⚠️  No EPS data found for {ticker}.")
    print(f"    💡 Set EPS_CSV_PATH to a CSV (date, eps) for full historical coverage.")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TTM PE CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_ttm_pe(prices: pd.Series, eps_quarterly: pd.Series) -> pd.Series:
    """
    Calculate daily Trailing Twelve-Month (TTM) P/E.

    TTM EPS at each date = sum of the 4 most recent quarterly EPS values
    whose report date is ≤ that date.  The TTM is then forward-filled
    between earnings dates so every trading day has a valid PE value.

    NaN is returned for dates with fewer than 4 quarters of history or
    where the TTM EPS is negative / zero (P/E undefined for loss-makers).
    """
    eps_q = eps_quarterly.sort_index()

    # Build TTM EPS series at each quarterly announcement date
    ttm_map: dict = {}
    for i in range(len(eps_q)):
        # Accumulate last 4 quarters up to and including quarter i
        start_i = max(0, i - 3)
        window  = eps_q.iloc[start_i : i + 1]
        if len(window) == 4:
            ttm = window.sum()
            if ttm > 0:                     # undefined for loss-making periods
                ttm_map[eps_q.index[i]] = ttm

    if not ttm_map:
        return pd.Series(dtype=float, index=prices.index)

    ttm_series = pd.Series(ttm_map).sort_index()

    # Forward-fill TTM EPS to every trading day in the price series
    daily_ttm = (ttm_series
                 .reindex(ttm_series.index.union(prices.index))
                 .sort_index()
                 .ffill()                   # carry last known TTM forward
                 .reindex(prices.index))

    pe = prices / daily_ttm
    pe[daily_ttm.isna()] = np.nan          # no TTM available yet
    return pe.rename(f"pe_{prices.name or 'ticker'}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SPREAD / RATIO COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_pe_comparison(pe_a: pd.Series, pe_b: pd.Series) -> pd.DataFrame:
    """
    Align both PE series to their common valid dates and compute:
      •  PE_A, PE_B            (raw trailing PE)
      •  ratio = PE_A / PE_B   (relative valuation)
      •  SMA for each window in SMA_WINDOWS applied to the ratio
    """
    df = pd.DataFrame({"pe_a": pe_a, "pe_b": pe_b}).dropna()
    if df.empty:
        raise ValueError("No overlapping valid P/E dates — check tickers or date range.")

    df["ratio"] = df["pe_a"] / df["pe_b"]

    for window, _, _ in SMA_WINDOWS:
        df[f"sma_{window}"] = (
            df["ratio"].rolling(window=window, min_periods=window).mean()
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PLOTLY CHART
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_pe(v: float) -> str:
    """Format a P/E value for display."""
    if np.isnan(v): return "—"
    return f"{v:.1f}×"

def _fmt_ratio(v: float) -> str:
    """Format a PE ratio value for display."""
    if np.isnan(v): return "—"
    pct = (v - 1) * 100
    sign = "+" if pct >= 0 else ""
    return f"{v:.3f}×  ({sign}{pct:.1f}%)"


def _inject_crosshair_js(html: str) -> str:
    # Inject a smooth SVG horizontal crosshair (same approach as zscore_spread.py).
    script = (
        '<script>\n'
        '(function () {\n'
        '  function attachCrosshair(gd) {\n'
        '    var plotSvg = gd.querySelector(".main-svg");\n'
        '    if (!plotSvg) return;\n'
        '    var hline = document.createElementNS("http://www.w3.org/2000/svg","line");\n'
        '    hline.setAttribute("stroke","rgba(255,255,255,0.28)");\n'
        '    hline.setAttribute("stroke-width","1");\n'
        '    hline.setAttribute("stroke-dasharray","6,4");\n'
        '    hline.setAttribute("pointer-events","none");\n'
        '    hline.style.display = "none";\n'
        '    plotSvg.appendChild(hline);\n'
        '    function update(evt) {\n'
        '      var L = gd._fullLayout;\n'
        '      if (!L) return;\n'
        '      var r = plotSvg.getBoundingClientRect();\n'
        '      var my = evt.clientY - r.top;\n'
        '      if (my < L.margin.t || my > L.height - L.margin.b) {\n'
        '        hline.style.display = "none"; return;\n'
        '      }\n'
        '      hline.setAttribute("x1", L.margin.l);\n'
        '      hline.setAttribute("x2", L.width - L.margin.r);\n'
        '      hline.setAttribute("y1", my);\n'
        '      hline.setAttribute("y2", my);\n'
        '      hline.style.display = "";\n'
        '    }\n'
        '    gd.addEventListener("mousemove", update);\n'
        '    gd.addEventListener("mouseleave", function(){ hline.style.display="none"; });\n'
        '  }\n'
        '  function tryAttach() {\n'
        '    document.querySelectorAll(".js-plotly-plot").forEach(function(gd) {\n'
        '      if (gd._crosshairAttached) return;\n'
        '      gd._crosshairAttached = true;\n'
        '      if (gd._fullLayout) { attachCrosshair(gd); }\n'
        '      else { gd.addEventListener("plotly_afterplot", function(){ attachCrosshair(gd); }, {once:true}); }\n'
        '    });\n'
        '  }\n'
        '  if (document.readyState==="loading") { document.addEventListener("DOMContentLoaded",tryAttach); }\n'
        '  else { tryAttach(); }\n'
        '  setTimeout(tryAttach, 600);\n'
        '})();\n'
        '</script>\n'
    )
    return html.replace("</body>", script + "</body>", 1)


def build_chart(
    df        : pd.DataFrame,
    ticker_a  : str,
    ticker_b  : str,
    ccy_label : str,
) -> go.Figure:
    """
    Two-panel interactive Plotly chart.

    Panel 1 (top, 38 %):  Trailing P/E for both stocks.
    Panel 2 (bottom, 62 %):  PE ratio (A / B) with SMA overlays and
                              reference levels.

    Hover: single master trace at parity (y=1) gives a unified popup
    listing both raw PEs, the ratio, and all SMA values.
    """
    dates  = df.index
    pe_a   = df["pe_a"].values
    pe_b   = df["pe_b"].values
    ratio  = df["ratio"].values
    n      = len(df)

    # ── y-axis ranges ─────────────────────────────────────────────────────────
    valid_pe  = np.concatenate([pe_a[~np.isnan(pe_a)], pe_b[~np.isnan(pe_b)]])
    valid_pe  = valid_pe[valid_pe > 0]
    pe_lo     = np.percentile(valid_pe, 2)    # ignore bottom 2% outliers
    pe_hi     = np.percentile(valid_pe, 98)   # ignore top 2% outliers
    pe_pad    = max(1.0, (pe_hi - pe_lo) * 0.08)
    pe_min    = float(PE_Y_MIN) if PE_Y_MIN is not None else max(0, pe_lo - pe_pad)
    pe_max    = float(PE_Y_MAX) if PE_Y_MAX is not None else pe_hi + pe_pad

    valid_r  = ratio[~np.isnan(ratio)]
    r_range  = valid_r.max() - valid_r.min()
    r_pad    = max(0.05, r_range * 0.06)
    r_min    = float(RATIO_Y_MIN) if RATIO_Y_MIN is not None else valid_r.min() - r_pad
    r_max    = float(RATIO_Y_MAX) if RATIO_Y_MAX is not None else valid_r.max() + r_pad

    # Substitute actual ticker names into reference level labels
    ref_levels = [
        (y, lbl.replace("A", ticker_a).replace("B", ticker_b), col, dash)
        for y, lbl, col, dash in REFERENCE_LEVELS
    ]

    # ── pre-build hover text for two master traces ─────────────────────────────
    # Row-1 (top panel): date + individual P/E values → drives vertical line in panel 1
    # Row-2 (bottom panel): separator + ratio + SMAs only (avoids duplicating date/PE)
    top_hover, bot_hover = [], []
    for i in range(n):
        d = pd.Timestamp(dates[i])
        top_hover.append("<br>".join([
            f"<b>{d.strftime('%Y-%m-%d')}</b>",
            f"<span style='color:#00BFFF'><b>{ticker_a} Trailing P/E</b></span>:  {_fmt_pe(pe_a[i])}",
            f"<span style='color:#FF8C00'><b>{ticker_b} Trailing P/E</b></span>:  {_fmt_pe(pe_b[i])}",
        ]))
        bot_rows = [
            "─" * 32,
            f"<b>PE Ratio  ({ticker_a} / {ticker_b})</b>:  {_fmt_ratio(ratio[i])}",
            "─" * 32,
        ]
        for window, label, colour in SMA_WINDOWS:
            v = df[f"sma_{window}"].iloc[i]
            bot_rows.append(
                f"<span style='color:{colour}'><b>{label}</b></span>:  {_fmt_ratio(v)}"
            )
        bot_hover.append("<br>".join(bot_rows))

    # ── figure: 2 rows, shared x-axis ─────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.38, 0.62],
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[
            f"Trailing P/E  — {ticker_a} vs {ticker_b}",
            f"PE Ratio  ({ticker_a} / {ticker_b})",
        ],
    )

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 1 — Individual Trailing P/E
    # ──────────────────────────────────────────────────────────────────────────
    COLOUR_A = "#00BFFF"   # sky blue  → Ticker A
    COLOUR_B = "#FF8C00"   # dark orange → Ticker B

    fig.add_trace(go.Scatter(
        x=dates, y=pe_a,
        mode="lines", name=f"{ticker_a} P/E",
        line=dict(color=COLOUR_A, width=1.8),
        hoverinfo="skip",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=pe_b,
        mode="lines", name=f"{ticker_b} P/E",
        line=dict(color=COLOUR_B, width=1.8),
        hoverinfo="skip",
    ), row=1, col=1)

    # Master hover for top panel — invisible line at geometric mean of PE values;
    # drives the vertical cursor line and popup when hovering over row 1.
    pe_a_s  = pd.Series(pe_a, index=dates)
    pe_b_s  = pd.Series(pe_b, index=dates)
    pe_mid  = np.sqrt(pe_a_s.fillna(pe_b_s) * pe_b_s.fillna(pe_a_s)).fillna(0).values
    fig.add_trace(go.Scatter(
        x=dates, y=pe_mid,
        mode="lines",
        line=dict(width=0, color="rgba(0,0,0,0)"),
        showlegend=False,
        hovertemplate="%{customdata}<extra></extra>",
        customdata=top_hover,
        name="_hover_top",
    ), row=1, col=1)

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 2 — PE Ratio with shaded bands, reference lines, SMAs
    # ──────────────────────────────────────────────────────────────────────────

    # Shaded bands — must use add_shape with yref="y2" for subplot row 2
    band_regions = [
        (0.67, 1.50, "rgba(255,255,255,0.03)"),  # fair value zone
        (1.50, 2.00, "rgba(255,165,  0, 0.07)"),  # A premium zone
        (0.50, 0.67, "rgba(255,165,  0, 0.07)"),  # A discount zone
        (2.00, r_max, "rgba(255, 68, 68, 0.10)"),  # A extreme premium
        (r_min, 0.50, "rgba(255, 68, 68, 0.10)"),  # A extreme discount
    ]
    for y0, y1, band_colour in band_regions:
        fig.add_shape(type="rect",
                      xref="paper", yref="y2",
                      x0=0, x1=1, y0=y0, y1=y1,
                      fillcolor=band_colour, layer="below", line_width=0)

    # Reference lines — explicit shapes + annotations bound to y2
    for y_val, label, colour, dash in ref_levels:
        lw = 1.4 if dash == "solid" else 1.0
        fig.add_shape(type="line",
                      xref="paper", yref="y2",
                      x0=0, x1=1, y0=y_val, y1=y_val,
                      line=dict(color=colour, width=lw, dash=dash))
        fig.add_annotation(
            xref="paper", yref="y2",
            x=1.002, y=y_val,
            text=f" {label}",
            showarrow=False,
            font=dict(color=colour, size=9),
            xanchor="left", yanchor="middle",
            # clip=False ensures text renders in the right-margin area
        )

    # Raw ratio (semi-transparent background)
    fig.add_trace(go.Scatter(
        x=dates, y=ratio,
        mode="lines",
        name="Raw PE Ratio",
        line=dict(color="rgba(200,200,210,0.22)", width=1),
        hoverinfo="skip",
    ), row=2, col=1)

    # SMA lines
    for rank, (window, label, colour) in enumerate(SMA_WINDOWS, start=1):
        col_name = f"sma_{window}"
        valid    = df[col_name].notna()
        fig.add_trace(go.Scatter(
            x=dates[valid], y=df[col_name][valid],
            mode="lines",
            name=label,
            line=dict(color=colour, width=2.2),
            legendrank=rank,
            hoverinfo="skip",
        ), row=2, col=1)

    # Master hover for bottom panel — ratio + SMAs only
    fig.add_trace(go.Scatter(
        x=dates,
        y=np.ones(n),
        mode="lines",
        line=dict(width=0, color="rgba(0,0,0,0)"),
        showlegend=False,
        hovertemplate="%{customdata}<extra></extra>",
        customdata=bot_hover,
        name="_hover_bot",
    ), row=2, col=1)

    # ──────────────────────────────────────────────────────────────────────────
    # LAYOUT
    # ──────────────────────────────────────────────────────────────────────────
    ccy_tag = f"  <span style='font-size:13px; color:#adbac7'>[{ccy_label}  — P/E is currency-neutral]</span>"

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        title=dict(
            text=(
                f"<b>{ticker_a} vs {ticker_b} — Relative P/E Comparison</b>"
                f"{ccy_tag}"
            ),
            font=dict(size=17, color="white"),
            x=0.5, xanchor="center",
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(22,27,34,0.88)",
            bordercolor="#30363d", borderwidth=1,
            font=dict(size=10, color="white"),
            tracegroupgap=8,
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#161b22", bordercolor="#30363d",
            font=dict(color="white", size=11),
            namelength=0,
        ),
        margin=dict(t=80, b=150, l=80, r=230),
        height=780,
    )

    # Shared x-axis (bottom panel only shows ticks)
    fig.update_xaxes(
        showgrid=True, gridcolor="#1e2530",
        linecolor="#30363d",
        tickfont=dict(color="#adbac7"),
        row=2, col=1,
    )
    fig.update_xaxes(showticklabels=False, row=1, col=1)

    # Panel 1 y-axis  (raw PE)
    fig.update_yaxes(
        title_text="Trailing P/E",
        range=[pe_min, pe_max],
        showgrid=True, gridcolor="#1e2530",
        minor=dict(showgrid=True, gridcolor="#161b22", gridwidth=0.5),
        linecolor="#30363d",
        tickfont=dict(color="#adbac7"),
        ticksuffix="×",
        row=1, col=1,
    )

    # Panel 2 y-axis  (PE ratio)
    fig.update_yaxes(
        title_text=f"PE Ratio  ({ticker_a} / {ticker_b})",
        range=[r_min, r_max],
        showgrid=True, gridcolor="#1e2530",
        minor=dict(showgrid=True, gridcolor="#161b22", gridwidth=0.5),
        linecolor="#30363d",
        tickfont=dict(color="#adbac7"),
        tickformat=".2f",
        ticksuffix="×",
        zeroline=False,
        row=2, col=1,
    )

    # Subplot title styling — only touch subplot-title annotations
    # (reference-level labels have yref="y2" and must keep their x=1.002)
    for ann in fig.layout.annotations:
        if not getattr(ann, "yref", None) or ann.yref == "paper":
            ann.update(font=dict(size=12, color="#adbac7"), x=0.5)

    # Range slider on shared x-axis
    fig.update_layout(
        xaxis2=dict(
            rangeslider=dict(visible=True, thickness=0.04,
                             bgcolor="#161b22", bordercolor="#30363d"),
        )
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def print_section(title: str) -> None:
    print(f"\n── {title} {'─' * max(0, 54 - len(title))}")


def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   📊  Relative P/E Ratio Comparison  📊                  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    end       = resolve_end_date(END_DATE)
    ccy_label = "CHF" if CONVERT_TO_CHF else "native"

    print(f"\n  Pair       : {TICKER_A}  vs  {TICKER_B}")
    print(f"  Date range : {START_DATE}  →  {end}")
    print(f"  Currency   : {ccy_label}  (P/E is currency-neutral)")
    print(f"  Output dir : {OUTPUT_DIR.resolve()}/\n")

    # ── fetch prices ──────────────────────────────────────────────────────────
    print_section(f"Fetching {TICKER_A}")
    prices_a = fetch_prices(TICKER_A, START_DATE, end)

    print_section(f"Fetching {TICKER_B}")
    prices_b = fetch_prices(TICKER_B, START_DATE, end)

    if prices_a is None or prices_b is None:
        print("\n❌  Failed to load prices for one or both tickers — aborting.")
        sys.exit(1)

    # ── fetch quarterly EPS ───────────────────────────────────────────────────
    print_section(f"EPS data [{TICKER_A}]")
    eps_a = get_quarterly_eps(TICKER_A)

    print_section(f"EPS data [{TICKER_B}]")
    eps_b = get_quarterly_eps(TICKER_B)

    if eps_a is None or eps_b is None:
        print("\n❌  EPS data unavailable for one or both tickers.")
        print("    Possible causes:")
        print("    • Network access blocked (sandbox / firewall)")
        print("    • Ticker not supported by yfinance for fundamentals")
        print("    • Non-US stock with limited Yahoo Finance coverage")
        sys.exit(1)

    # ── compute TTM PE ────────────────────────────────────────────────────────
    print_section("Computing TTM P/E")
    prices_a.name = TICKER_A
    prices_b.name = TICKER_B

    pe_a = compute_ttm_pe(prices_a, eps_a)
    pe_b = compute_ttm_pe(prices_b, eps_b)

    valid_a = pe_a.notna().sum()
    valid_b = pe_b.notna().sum()
    print(f"  {TICKER_A}: {valid_a} days with valid P/E  "
          f"(range: {pe_a.min():.1f}× – {pe_a.max():.1f}×)")
    print(f"  {TICKER_B}: {valid_b} days with valid P/E  "
          f"(range: {pe_b.min():.1f}× – {pe_b.max():.1f}×)")

    if valid_a < 100 or valid_b < 100:
        print("  ⚠️  Very few valid P/E data points — results may be unreliable.")

    # ── ratio & SMAs ──────────────────────────────────────────────────────────
    print_section("Computing PE ratio & SMAs")
    try:
        df = compute_pe_comparison(pe_a, pe_b)
    except ValueError as exc:
        print(f"\n❌  {exc}")
        sys.exit(1)

    overlap = len(df)
    print(f"  Overlapping valid PE days: {overlap}  "
          f"({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})")

    if overlap < max(w for w, *_ in SMA_WINDOWS):
        print(f"  ⚠️  Overlap shorter than longest SMA ({max(w for w, *_ in SMA_WINDOWS)} days)"
              f" — some SMAs will be partial.")

    cur_ratio = df["ratio"].iloc[-1]
    print(f"\n  Current PE  {TICKER_A}: {_fmt_pe(df['pe_a'].iloc[-1])}")
    print(f"  Current PE  {TICKER_B}: {_fmt_pe(df['pe_b'].iloc[-1])}")
    print(f"  Current ratio ({TICKER_A}/{TICKER_B}): {_fmt_ratio(cur_ratio)}")
    for window, label, _ in SMA_WINDOWS:
        col = f"sma_{window}"
        v   = df[col].dropna().iloc[-1] if df[col].notna().any() else float("nan")
        print(f"  {label:<42} {_fmt_ratio(v)}")

    # ── chart ─────────────────────────────────────────────────────────────────
    print_section("Building interactive chart")
    fig = build_chart(df, TICKER_A, TICKER_B, ccy_label)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname   = f"{TICKER_A}_vs_{TICKER_B}_PE_ratio.html"
    outpath = OUTPUT_DIR / fname
    fig.write_html(
        str(outpath),
        include_plotlyjs="cdn",
        config={
            "scrollZoom"    : True,
            "displayModeBar": True,
            "toImageButtonOptions": {
                "format"  : "png",
                "filename": fname.replace(".html", ""),
                "height"  : 900,
                "width"   : 1600,
                "scale"   : 2,
            },
        },
    )

    html_text = outpath.read_text(encoding="utf-8")
    outpath.write_text(_inject_crosshair_js(html_text), encoding="utf-8")

    print(f"  💾 Interactive chart → {outpath.resolve()}")
    print(f"\n{'─' * 60}")
    print(f"  Done.  Open {fname} in any browser.")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()
