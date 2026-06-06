"""
==============================================================================
  Relative Z-Score Spread Analyser
  ----------------------------------
  Compares two assets by computing their individual Log-Power-Law regression
  Z-scores and charting the daily spread (Z_A − Z_B) together with three
  moving-average baselines and structural reference levels.

  Model  :  ln(Price) = a · ln(t) + b    (OLS power-law regression)
  Z-Score:  Z = (ln(Price) − ln(Predicted)) / sigma_lifetime
  Spread :  Z_Spread = Z_A − Z_B

  When the spread is POSITIVE  → Ticker A is relatively overvalued vs B
  When the spread is NEGATIVE  → Ticker A is relatively undervalued vs B

  Usage:
      python zscore_spread.py

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
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  —  edit these values, then just run: python zscore_spread.py
# ─────────────────────────────────────────────────────────────────────────────

TICKER_A   = "AAPL"          # <- left-hand ticker of the spread (A − B)
TICKER_B   = "MSFT"          # <- right-hand ticker
START_DATE = "2012-01-01"    # <- YYYY-MM-DD  (use a long history for stable regression)
END_DATE   = "today"         # <- YYYY-MM-DD  or  "today"

# ── Currency conversion ───────────────────────────────────────────────────────
# When True both prices are converted to CHF via daily FX rates before
# regression.  Works for USD, EUR, GBP, JPY … any Yahoo-quoted pair.
CONVERT_TO_CHF = True   # <- True | False

# ── Moving average windows ────────────────────────────────────────────────────
# Each entry: (window_days, label, hex_colour)
# The label is shown in the legend and hover tooltip.
SMA_WINDOWS = [
    ( 50, "50-day SMA  (Fast / Tactical)",    "#FFD700"),  # gold
    (200, "200-day SMA (Medium-Term Trend)",   "#00BFFF"),  # sky blue
    (600, "600-day SMA (Slow / Structural)",   "#FF69B4"),  # pink
]

# ── Structural reference levels ───────────────────────────────────────────────
# Each entry: (y_value, label, colour, plotly_dash_style)
REFERENCE_LEVELS = [
    ( 0.0, "Equilibrium  (Z=0)",             "#888888", "solid"),
    (+1.5, "Upper Bounds  (+1.5σ)",          "#FFA500", "dash"),
    (-1.5, "Lower Bounds  (−1.5σ)",          "#FFA500", "dash"),
    (+3.0, "Upper Extreme  (+3.0σ)",         "#FF4444", "dash"),
    (-3.0, "Lower Extreme  (−3.0σ)",         "#FF4444", "dash"),
]

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("zscore_charts")

# ── Y-axis limits ────────────────────────────────────────────────────────────
# The y-axis normally auto-fits to the actual spread range (+ a small buffer).
# Set these to clip extreme early-history swings and zoom into recent action.
# Set either (or both) to None to keep the automatic data-driven value.
Y_AXIS_MIN = None   # e.g. -4.0  |  None = automatic
Y_AXIS_MAX = None   # e.g. +4.0  |  None = automatic


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA FETCHING & CHF CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def resolve_end_date(end: str) -> str:
    return datetime.today().strftime("%Y-%m-%d") if end.strip().lower() == "today" else end


def fetch_prices(ticker: str, start: str, end: str) -> "pd.Series | None":
    """Download Adjusted Close prices; return a DatetimeIndex Series or None."""
    print(f"  ↓ {ticker} prices …", end=" ", flush=True)
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    except Exception as exc:
        print(f"FAILED ({exc})")
        return None
    if df is None or df.empty:
        print("FAILED (no data — check ticker or date range)")
        return None
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)
    col = "Close" if "Close" in df.columns else df.columns[0]
    s   = df[col].dropna().astype(float)
    if len(s) < 60:
        print(f"FAILED (only {len(s)} data points)")
        return None
    print(f"OK  ({len(s)} trading days)")
    return s


def get_native_currency(ticker: str) -> str:
    """Ask yfinance for the currency a stock is quoted in; fallback to USD."""
    try:
        info = yf.Ticker(ticker).fast_info
        return str(info.get("currency") or info.get("Currency") or "USD").upper()
    except Exception:
        return "USD"


def fetch_chf_rates(
    currency: str,
    index   : pd.DatetimeIndex,
    start   : str,
    end     : str,
) -> "pd.Series | None":
    """
    Download daily <currency>/CHF exchange rates and align them to `index`
    via forward-fill (then back-fill for leading gaps).
    Returns None if currency is already CHF or download fails.
    """
    if currency == "CHF":
        return None
    fx_ticker = f"{currency}CHF=X"
    print(f"  ↓ {currency}/CHF rates ({fx_ticker}) …", end=" ", flush=True)
    try:
        fx = yf.download(fx_ticker, start=start, end=end, progress=False, auto_adjust=True)
    except Exception as exc:
        print(f"FAILED ({exc})")
        return None
    if fx is None or fx.empty:
        print("FAILED (no data)")
        return None
    if hasattr(fx.columns, "levels"):
        fx.columns = fx.columns.get_level_values(0)
    col = "Close" if "Close" in fx.columns else fx.columns[0]
    fx_s    = fx[col].dropna().astype(float)
    aligned = (fx_s
               .reindex(fx_s.index.union(index))
               .ffill()
               .reindex(index)
               .bfill()
               .fillna(1.0))
    print(f"OK  (avg {aligned.mean():.4f}  min {aligned.min():.4f}  max {aligned.max():.4f})")
    return aligned


def load_ticker_prices(ticker: str, start: str, end: str) -> "pd.Series | None":
    """Fetch prices and optionally convert to CHF.  Returns Series or None."""
    prices = fetch_prices(ticker, start, end)
    if prices is None:
        return None
    if CONVERT_TO_CHF:
        ccy   = get_native_currency(ticker)
        rates = fetch_chf_rates(ccy, prices.index, start, end)
        if rates is not None:
            prices = prices * rates
        else:
            print(f"    ⚠️  CHF conversion unavailable for {ticker} ({ccy}) "
                  f"— keeping native currency.")
    return prices


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LOG-POWER-LAW REGRESSION & Z-SCORE
# ─────────────────────────────────────────────────────────────────────────────

def compute_zscore(
    prices: pd.Series,
    ticker: str,
) -> "tuple[pd.Series, float, float, float]":
    """
    Fit  ln(Price) = a · ln(t) + b  via OLS where t = 1, 2, …, N.

    Returns
    -------
    z_scores : pd.Series   daily Z-scores on prices.index
    a, b     : float       OLS slope and intercept
    sigma    : float       lifetime std-dev of log-price residuals
    """
    t         = np.arange(1, len(prices) + 1, dtype=float)
    ln_t      = np.log(t)
    ln_price  = np.log(prices.values.astype(float))

    a, b, r_val, _, _ = stats.linregress(ln_t, ln_price)

    predicted = a * ln_t + b
    residuals = ln_price - predicted
    sigma     = residuals.std(ddof=2)          # unbiased estimator
    z_scores  = pd.Series(residuals / sigma, index=prices.index, name=f"z_{ticker}")

    r2 = r_val ** 2
    print(f"    ln(P) = {a:.4f}·ln(t) + {b:.4f}  |  R²={r2:.4f}  σ={sigma:.4f}")
    if r2 < 0.50:
        print(f"    ⚠️  Low R² ({r2:.3f}) — power-law may not describe {ticker} "
              f"well in this date range.  Results shown for reference.")
    return z_scores, a, b, sigma


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SPREAD MODELLING
# ─────────────────────────────────────────────────────────────────────────────

def compute_spread(z_a: pd.Series, z_b: pd.Series) -> pd.DataFrame:
    """
    Align both Z-score series to their common overlapping dates, then compute:
      •  Raw daily Z-spread  = Z_A − Z_B
      •  SMA for each window defined in SMA_WINDOWS
    """
    df = pd.DataFrame({"z_a": z_a, "z_b": z_b}).dropna()
    if df.empty:
        raise ValueError("No overlapping dates between the two tickers — "
                         "check START_DATE or ticker validity.")
    df["spread"] = df["z_a"] - df["z_b"]
    for window, _, _ in SMA_WINDOWS:
        df[f"sma_{window}"] = (
            df["spread"].rolling(window=window, min_periods=window).mean()
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4.  INTERACTIVE PLOTLY CHART
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_z(z: "float | None") -> str:
    """Format a Z-score value for display."""
    if z is None or (isinstance(z, float) and np.isnan(z)):
        return "—"
    sign = "+" if z >= 0 else ""
    return f"{sign}{z:.3f}σ"


def build_chart(
    df      : pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    reg_a   : tuple,   # (a, b, sigma)
    reg_b   : tuple,
) -> go.Figure:
    """
    Construct the interactive Z-spread Plotly chart.

    Hover strategy (same as rainbow script):
    All individual traces carry hoverinfo="skip".  A single invisible line
    at y=0 holds pre-built customdata strings, giving full control over
    popup content and ordering.
    """
    a_a, b_a, sig_a = reg_a
    a_b, b_b, sig_b = reg_b
    ccy_label = "CHF" if CONVERT_TO_CHF else "native ccy"
    dates     = df.index
    spread    = df["spread"].values
    n         = len(df)

    # ── determine sensible y-axis range ──────────────────────────────────
    # Dynamic y-axis: fit tightly to actual spread range + a small
    # proportional buffer so the reference lines are never clipped.
    data_range = spread.max() - spread.min()
    y_pad      = max(0.3, data_range * 0.06)   # at least 0.3σ, else 6% of range
    y_min      = spread.min() - y_pad
    y_max      = spread.max() + y_pad
    # Optional hard overrides (e.g. to clip extreme early history)
    if Y_AXIS_MIN is not None:
        y_min = float(Y_AXIS_MIN)
    if Y_AXIS_MAX is not None:
        y_max = float(Y_AXIS_MAX)

    # ── pre-build master hover text (one string per date) ─────────────────
    master_hover = []
    for i in range(n):
        d    = pd.Timestamp(dates[i])
        rows = [f"<b>{d.strftime('%Y-%m-%d')}</b>"]
        rows.append(
            f"<b>Z-Spread  ({ticker_a} − {ticker_b})</b>:  "
            f"{_fmt_z(spread[i])}"
        )
        rows.append("─" * 32)
        for window, label, colour in SMA_WINDOWS:
            v = df[f"sma_{window}"].iloc[i]
            rows.append(
                f"<span style='color:{colour}'><b>{label}</b></span>:  {_fmt_z(v)}"
            )
        rows.append("─" * 32)
        rows.append(
            f"<i style='color:#adbac7'>Positive spread → {ticker_a} relatively extended<br>"
            f"Negative spread → {ticker_b} relatively extended</i>"
        )
        master_hover.append("<br>".join(rows))

    # ── figure ────────────────────────────────────────────────────────────
    fig = go.Figure()

    # ── shaded reference bands (drawn first, behind all traces) ───────────
    #   ± 0–1.5σ  : neutral zone (very subtle white tint)
    #   ± 1.5–3σ  : institutional caution zone (amber tint)
    #   beyond ±3σ: extreme extension zone (red tint)
    band_regions = [
        (-1.5,  1.5,  "rgba(255,255,255,0.03)"),   # neutral equilibrium
        ( 1.5,  3.0,  "rgba(255,165,  0, 0.07)"),  # upper caution
        (-3.0, -1.5,  "rgba(255,165,  0, 0.07)"),  # lower caution
        ( 3.0,  y_max,"rgba(255, 68, 68, 0.10)"),  # upper extreme
        ( y_min,-3.0, "rgba(255, 68, 68, 0.10)"),  # lower extreme
    ]
    for y0, y1, colour in band_regions:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=colour, layer="below", line_width=0)

    # ── horizontal reference lines ─────────────────────────────────────────
    for y_val, label, colour, dash in REFERENCE_LEVELS:
        fig.add_hline(
            y=y_val,
            line=dict(color=colour, width=1.4 if dash == "solid" else 1.0, dash=dash),
            annotation_text=f"  {label}",
            annotation_position="top right",
            annotation_font=dict(color=colour, size=9),
            annotation_xanchor="left",
        )

    # ── raw Z-spread (semi-transparent so MAs stay readable) ──────────────
    fig.add_trace(go.Scatter(
        x=dates, y=spread,
        mode="lines",
        name="Raw Z-Spread (daily)",
        line=dict(color="rgba(200,200,210,0.22)", width=1),
        hoverinfo="skip",
    ))

    # ── SMA lines ─────────────────────────────────────────────────────────
    for rank, (window, label, colour) in enumerate(SMA_WINDOWS, start=1):
        col   = f"sma_{window}"
        valid = df[col].notna()
        fig.add_trace(go.Scatter(
            x=dates[valid], y=df[col][valid],
            mode="lines",
            name=label,
            line=dict(color=colour, width=2.2),
            legendrank=rank,
            hoverinfo="skip",
        ))

    # ── master hover trace ────────────────────────────────────────────────
    # Invisible line at y=0; sole source of hover tooltips.
    fig.add_trace(go.Scatter(
        x=dates,
        y=np.zeros(n),
        mode="lines",
        line=dict(width=0, color="rgba(0,0,0,0)"),
        showlegend=False,
        hovertemplate="%{customdata}<extra></extra>",
        customdata=master_hover,
        name="_hover",
    ))

    # ── layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        title=dict(
            text=(
                f"<b>{ticker_a} vs {ticker_b} — Relative Z-Score Spread</b>  "
                f"<span style='font-size:13px; color:#adbac7'>[{ccy_label}]</span><br>"
                f"<sup style='color:#adbac7'>"
                f"{ticker_a}: ln(P) = {a_a:.4f}·ln(t) + {b_a:.4f},  σ={sig_a:.4f}"
                f"  &nbsp;|&nbsp;  "
                f"{ticker_b}: ln(P) = {a_b:.4f}·ln(t) + {b_b:.4f},  σ={sig_b:.4f}"
                f"</sup>"
            ),
            font=dict(size=17, color="white"),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title=None,
            showgrid=True, gridcolor="#1e2530",
            linecolor="#30363d",
            tickfont=dict(color="#adbac7"),
            rangeslider=dict(
                visible=True, thickness=0.06,
                bgcolor="#161b22", bordercolor="#30363d",
            ),
        ),
        yaxis=dict(
            title=f"Z-Score Spread  ({ticker_a} − {ticker_b})",
            range=[y_min, y_max],
            showgrid=True, gridcolor="#1e2530",
            minor=dict(showgrid=True, gridcolor="#161b22", gridwidth=0.5),
            linecolor="#30363d",
            tickfont=dict(color="#adbac7"),
            ticksuffix="σ",
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
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
        # Extra right margin for hline annotations; extra bottom for legend
        margin=dict(t=100, b=190, l=80, r=200),
        height=720,
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def print_section(title: str) -> None:
    print(f"\n── {title} {'─' * max(0, 54 - len(title))}")


def _inject_crosshair_js(html: str) -> str:
    # Post-process Plotly HTML to add a smooth SVG horizontal crosshair
    # that follows the cursor in sync with Plotly's vertical hover line.
    # Uses direct SVG DOM manipulation (no Plotly.relayout calls) so it
    # is lag-free even on large datasets.
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


def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   📊  Relative Z-Score Spread Analyser  📊               ║")
    print("╚══════════════════════════════════════════════════════════╝")

    end   = resolve_end_date(END_DATE)
    start = START_DATE
    ccy   = "CHF" if CONVERT_TO_CHF else "native currency"

    print(f"\n  Pair       : {TICKER_A}  vs  {TICKER_B}")
    print(f"  Date range : {start}  →  {end}")
    print(f"  Currency   : {ccy}")
    print(f"  Output dir : {OUTPUT_DIR.resolve()}/\n")

    # ── fetch data ────────────────────────────────────────────────────────
    print_section(f"Fetching {TICKER_A}")
    prices_a = load_ticker_prices(TICKER_A, start, end)

    print_section(f"Fetching {TICKER_B}")
    prices_b = load_ticker_prices(TICKER_B, start, end)

    if prices_a is None or prices_b is None:
        print("\n❌  Failed to load one or both tickers — aborting.")
        sys.exit(1)

    # ── regression & Z-scores ─────────────────────────────────────────────
    print_section(f"Regression [{TICKER_A}]")
    z_a, a_a, b_a, sig_a = compute_zscore(prices_a, TICKER_A)

    print_section(f"Regression [{TICKER_B}]")
    z_b, a_b, b_b, sig_b = compute_zscore(prices_b, TICKER_B)

    # ── spread ────────────────────────────────────────────────────────────
    print_section("Computing Z-Spread & SMAs")
    try:
        df = compute_spread(z_a, z_b)
    except ValueError as exc:
        print(f"\n❌  {exc}")
        sys.exit(1)

    overlap_days = len(df)
    print(f"  Overlapping days : {overlap_days}  "
          f"({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})")
    if overlap_days < max(w for w, *_ in SMA_WINDOWS):
        print(f"  ⚠️  Overlap shorter than longest SMA window "
              f"({max(w for w, *_ in SMA_WINDOWS)} days) — some SMAs will be partial.")

    # Current readings summary
    print(f"\n  Current Z-Spread : {_fmt_z(df['spread'].iloc[-1])}")
    for window, label, _ in SMA_WINDOWS:
        col = f"sma_{window}"
        v   = df[col].dropna().iloc[-1] if df[col].notna().any() else float("nan")
        print(f"  {label:<42} {_fmt_z(v)}")

    # ── build & save chart ────────────────────────────────────────────────
    print_section("Building interactive chart")
    fig = build_chart(
        df, TICKER_A, TICKER_B,
        (a_a, b_a, sig_a),
        (a_b, b_b, sig_b),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname   = f"{TICKER_A}_vs_{TICKER_B}_zscore_spread.html"
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

    # Inject horizontal crosshair into the saved HTML
    html_text = outpath.read_text(encoding="utf-8")
    outpath.write_text(_inject_crosshair_js(html_text), encoding="utf-8")

    print(f"  💾 Interactive chart → {outpath.resolve()}")
    print(f"\n{'─' * 60}")
    print(f"  Done.  Open {fname} in any browser.")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()
