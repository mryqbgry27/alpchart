"""
==============================================================================
  Stock Rainbow Regression Chart Generator
  ----------------------------------------
  Fits a power-law (log-log) regression to historical Adjusted Close prices
  and draws colourful "rainbow bands" à la the Bitcoin Power-Law Rainbow Chart.

  Model:   ln(Price) = a * ln(day) + b   ←→   Price ~ e^b * day^a
  Bands:   ± 1σ, ±2σ, ±3σ offsets on the ln-price residuals produce bands
           that appear as parallel curves when viewed on a log-price axis.

  Usage:
      python rainbow_regression.py
      … then follow the prompts.

  Dependencies:
      pip install yfinance plotly numpy scipy pandas
==============================================================================
"""

import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")   # suppress yfinance/pandas deprecation noise


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONFIGURATION  –  band definitions (label, std-dev multiplier, colour)
# ─────────────────────────────────────────────────────────────────────────────

BANDS = [
    # (label,                             sigma_multiplier,  hex_colour)
    ("Structural Extension Floor",         -3.0,              "#d7191c"),
    ("Negative Compression Band",          -1.5,              "#fdae61"),
    ("Equilibrium Growth Trend",            0.0,              "#ffffbf"),
    ("Positive Expansion Band",             1.5,              "#a6d96a"),
    ("Structural Extension Ceiling",        3.0,              "#1a9641"),
]

# How many σ above/below each band centre to shade the filled area
BAND_HALF_WIDTH = 0.5   # σ units — keeps bands visually distinct without overlap

# ── Y-axis floor ──────────────────────────────────────────────────────────────
# Prices below this value are clipped from the chart view.  The regression is
# still fitted on ALL data; this only affects the visible viewport.
# Set to None to show the full range (including the steep early-history dip).
#   Examples:  0.10  →  hide anything below $0.10
#              1.00  →  hide anything below $1.00
#              None  →  no clipping
# ── Y-axis floor ────────────────────────────────────────────────────────────────────────────
# The y-axis bottom is set automatically to each stock's own minimum
# Adjusted Close multiplied by this buffer (< 1.0 so the lowest point
# sits a little above the chart floor with breathing room).
# 0.7 = floor 30% below the historical low; 0.5 = more room, 0.9 = less.
Y_FLOOR_BUFFER = 0.2   # ← edit this to taste (0.2 = floor at 20% of historical low)

# Chart output folder (created automatically)
OUTPUT_DIR = Path("rainbow_charts")

# ── Tickers & date range ─────────────────────────────────────────────────────
# Edit these three values and just run:  python rainbow_regression.py
# START_DATE: earlier = more data = more reliable fit (5-15 yrs ideal)
# END_DATE:   "today" always fetches up to the current date, or pin a
#             specific string like "2024-12-31" to freeze the range.
TICKERS    = ["AAPL", "MSFT", "GOOGL", "AMZN"]   # <- edit this list
START_DATE = "2012-01-01"                 # <- YYYY-MM-DD
END_DATE   = "today"                      # <- YYYY-MM-DD  or  "today"

# ── Currency conversion ──────────────────────────────────────────────────────
# When True, prices are multiplied by the daily <native-currency>/CHF rate
# fetched from Yahoo Finance.  Works for any stock currency (USD, EUR, GBP …).
# Set to False to keep prices in the stock's native currency.
CONVERT_TO_CHF = True   # <- True | False

# ── Moving average overlays ──────────────────────────────────────────────────
# List any number of moving averages to overlay on each chart.
# Each entry:  (window_days, "SMA" or "EMA", hex_colour)
# Set to an empty list [] to disable all MAs.
MOVING_AVERAGES = [
    (200,  "SMA", "#00bfff"),   # 200-day SMA — ice blue
    (600,  "SMA", "#ff69b4"),   # 600-day SMA — pink
]

# ── Forecast extension ───────────────────────────────────────────────────────
# How many months ahead to project the regression bands beyond the last
# data point.  One calendar month ≈ 30 days.  Set to 0 to disable.
FORECAST_MONTHS = 6   # <- edit this (e.g. 3, 6, 12, 24)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_price_data(ticker: str, start: str, end: str) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Download Adjusted Close prices from Yahoo Finance.

    Returns
    -------
    dates  : np.ndarray of datetime64
    prices : np.ndarray of float
    or None if the download failed / returned empty data.
    """
    print(f"  ↓ Fetching data for {ticker} …", end=" ", flush=True)
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    except Exception as exc:
        print(f"FAILED\n    Error: {exc}")
        return None

    if df is None or df.empty:
        print("FAILED\n    No data returned — check the ticker or date range.")
        return None

    # Flatten MultiIndex columns that yfinance sometimes returns
    if isinstance(df.columns, type(df.columns)) and hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)

    col = "Close" if "Close" in df.columns else df.columns[0]
    prices = df[col].dropna().values.astype(float)
    dates  = df[col].dropna().index.values   # numpy datetime64

    if len(prices) < 30:
        print(f"FAILED\n    Only {len(prices)} data-points — need at least 30.")
        return None

    print(f"OK  ({len(prices)} trading days)")
    return dates, prices


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 3b. CHF CONVERSION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_native_currency(ticker: str) -> str:
    """
    Ask yfinance for the currency the stock is quoted in (e.g. "USD", "EUR").
    Falls back to "USD" if the metadata is unavailable.
    """
    try:
        info = yf.Ticker(ticker).fast_info
        ccy  = info.get("currency") or info.get("Currency") or "USD"
        return str(ccy).upper()
    except Exception:
        return "USD"


def fetch_chf_rates(
    currency: str,
    dates   : "np.ndarray",
    start   : str,
    end     : str,
) -> "np.ndarray | None":
    """
    Download daily <currency>/CHF exchange rates and align them to the
    stock's exact trading dates via forward-fill (then back-fill for any
    leading gaps at the very start of the range).

    Returns
    -------
    rates : np.ndarray aligned to `dates`, or None on failure.
            Multiplying the price array by this gives CHF prices.
    """
    import pandas as pd

    if currency == "CHF":
        return None   # already in CHF — no conversion needed

    fx_ticker = f"{currency}CHF=X"
    print(f"  ↓ Fetching {currency}/CHF rates ({fx_ticker}) …", end=" ", flush=True)
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
    fx_series = fx[col].dropna()

    # Re-index to stock trading dates, forward-fill gaps, then back-fill
    # any leading NaNs at the start, and finally fill remaining with 1.0.
    stock_dates = pd.DatetimeIndex(dates.astype("datetime64[D]"))
    fx_aligned  = (fx_series
                   .reindex(fx_series.index.union(stock_dates))
                   .ffill()
                   .reindex(stock_dates)
                   .bfill()
                   .fillna(1.0))

    if fx_aligned.isna().all():
        print("FAILED (could not align dates)")
        return None

    rates = fx_aligned.values.astype(float)
    print(f"OK  (avg: {rates.mean():.4f}  "
          f"min: {rates.min():.4f}  max: {rates.max():.4f})")
    return rates


# 4.  LOGARITHMIC REGRESSION  (power-law fit in log-log space)
# ─────────────────────────────────────────────────────────────────────────────

def fit_log_regression(prices: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """
    Fit:   ln(Price[t]) = a * ln(t) + b
    where  t = 1, 2, …, N  (trading-day index, 1-based to avoid ln(0))

    Returns
    -------
    a         : slope in log-log space  (growth exponent)
    b         : intercept
    sigma     : standard deviation of residuals (ln-price units)
    residuals : array of ln(Price) - fitted_ln(Price)
    """
    n = len(prices)
    t = np.arange(1, n + 1, dtype=float)   # 1-based day index

    ln_t     = np.log(t)
    ln_price = np.log(prices)

    # Ordinary Least Squares via scipy.stats.linregress
    slope, intercept, r_value, p_value, se = stats.linregress(ln_t, ln_price)

    fitted    = slope * ln_t + intercept
    residuals = ln_price - fitted
    sigma     = residuals.std(ddof=2)   # unbiased estimate

    print(f"    Regression: ln(P) = {slope:.4f}·ln(t) + {intercept:.4f}  "
          f"| R²={r_value**2:.4f}  σ={sigma:.4f}")

    return slope, intercept, sigma, residuals


# ─────────────────────────────────────────────────────────────────────────────
# 5.  BAND COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_bands(
    a: float,
    b: float,
    sigma: float,
    t_full: np.ndarray,
) -> list[dict]:
    """
    For each band definition compute the centre curve and the upper/lower
    edges of the filled region, all in price-space (exp of the ln-model).

    Parameters
    ----------
    t_full : extended day indices (includes a short forecast window)

    Returns a list of dicts:
        {label, colour, centre, lower, upper}
    """
    ln_t    = np.log(t_full)
    baseline = a * ln_t + b     # ln-price values on the regression line

    result = []
    for label, sigma_mult, colour in BANDS:
        # Centre of this band in ln-price space
        centre_ln = baseline + sigma_mult * sigma

        # Thin filled region around the centre (for visual clarity)
        lower_ln  = centre_ln - BAND_HALF_WIDTH * sigma
        upper_ln  = centre_ln + BAND_HALF_WIDTH * sigma

        result.append({
            "label"     : label,
            "colour"    : colour,
            "sigma_mult": sigma_mult,   # stored so the legend can display it
            "centre"    : np.exp(centre_ln),
            "lower"     : np.exp(lower_ln),
            "upper"     : np.exp(upper_ln),
        })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def _hex_to_rgba(hex_colour: str, alpha: float = 1.0) -> str:
    """Convert a #rrggbb hex string to a CSS rgba() string for Plotly."""
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def make_rainbow_chart(
    ticker        : str,
    dates         : np.ndarray,
    prices        : np.ndarray,
    a             : float,
    b             : float,
    sigma         : float,
    save_dir      : Path,
    price_currency: str = "USD",
) -> None:
    """
    Build an interactive Plotly rainbow regression chart and save it as HTML.

    Hover strategy
    --------------
    All individual traces have hoverinfo="skip".  A single invisible master
    trace carries a pre-built customdata string per date that lists bands in
    Ceiling→Floor order (matching the visual top-to-bottom chart layout),
    followed by the price and any MAs.  This sidesteps Plotly's default
    behaviour of sorting unified-hover entries by y-value (low→high).
    """

    n             = len(prices)
    FORECAST_DAYS = max(1, int(FORECAST_MONTHS * 30))
    t_full        = np.arange(1, n + FORECAST_DAYS + 1, dtype=float)

    # ── date arrays ───────────────────────────────────────────────────────────
    hist_dates     = pd.to_datetime(dates.astype("datetime64[D]"))
    last_date      = hist_dates[-1]
    forecast_dates = pd.date_range(
        last_date + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq="D"
    )
    all_dates = hist_dates.append(forecast_dates)
    n_all     = len(all_dates)

    bands    = compute_bands(a, b, sigma, t_full)
    sym      = "CHF " if price_currency == "CHF" else "$"
    ccy_note = f" [{price_currency}]" if CONVERT_TO_CHF else ""

    def fmt(p: float) -> str:
        if p >= 1_000_000: return f"{sym}{p/1_000_000:.2f}M"
        if p >= 1_000:     return f"{sym}{p:,.0f}"
        if p >= 1:         return f"{sym}{p:.2f}"
        return f"{sym}{p:.4f}"

    # ── pre-compute MAs (must happen before hover text is built) ─────────────
    price_series = pd.Series(prices.astype(float), index=hist_dates)
    ma_list = []   # (label, colour, pd.Series on hist_dates)
    for (window, ma_type, ma_colour) in MOVING_AVERAGES:
        if window > n:
            print(f"    ⚠️  {ma_type}-{window} skipped — fewer than {window} data points.")
            continue
        if ma_type.upper() == "EMA":
            ma_vals  = price_series.ewm(span=window, adjust=False).mean()
            ma_label = f"{window}-day EMA"
        else:
            ma_vals  = price_series.rolling(window=window, min_periods=window).mean()
            ma_label = f"{window}-day SMA"
        ma_list.append((ma_label, ma_colour, ma_vals))

    # ── build master hover strings (one per date, Ceiling→Floor) ─────────────
    # Using reversed(bands) keeps the topmost band first in the tooltip,
    # matching the visual order on the chart.
    master_hover = []
    for i in range(n_all):
        d       = pd.Timestamp(all_dates[i])
        is_hist = i < n
        rows    = [f"<b>{d.strftime('%Y-%m-%d')}</b>"]
        for band in reversed(bands):          # Ceiling first → Floor last
            stag = f"({band['sigma_mult']:+.1f}σ)"
            rows.append(
                f"<span style='color:{band["colour"]}'>■</span> "
                f"<b>{band['label']}</b> {stag} — "
                f"centre {fmt(band['centre'][i])}  "
                f"<i>({fmt(band['lower'][i])}–{fmt(band['upper'][i])})</i>"
            )
        if is_hist:
            rows.append("─" * 34)
            rows.append(f"<b>{ticker} Close</b> — {fmt(prices[i])}")
            for ma_label, _, ma_vals in ma_list:
                v = ma_vals.iloc[i]
                if pd.notna(v):
                    rows.append(f"<b>{ma_label}</b> — {fmt(v)}")
        master_hover.append("<br>".join(rows))

    # ── figure ────────────────────────────────────────────────────────────────
    fig = go.Figure()

    # ── rainbow bands (hoverinfo="skip" — master trace handles tooltips) ──────
    for band_idx, band in enumerate(reversed(bands)):
        band_rank  = len(bands) - band_idx    # Floor=1 … Ceiling=5 in legend
        sigma_tag  = f"({band['sigma_mult']:+.1f}σ)"
        full_label = f"{band['label']}  {sigma_tag}"
        rgba_fill  = _hex_to_rgba(band["colour"], alpha=0.28)
        rgba_line  = _hex_to_rgba(band["colour"], alpha=0.85)

        # Lower boundary — invisible anchor for tonexty fill
        fig.add_trace(go.Scatter(
            x=all_dates, y=band["lower"],
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
            name=full_label + "_lo",
        ))
        # Upper boundary — fills back down to the lower trace
        fig.add_trace(go.Scatter(
            x=all_dates, y=band["upper"],
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=rgba_fill,
            showlegend=True, legendgroup=full_label,
            name=full_label, legendrank=band_rank,
            hoverinfo="skip",
        ))
        # Dashed centre line
        fig.add_trace(go.Scatter(
            x=all_dates, y=band["centre"],
            mode="lines",
            line=dict(width=1.2, color=rgba_line, dash="dash"),
            showlegend=False, legendgroup=full_label,
            hoverinfo="skip",
            name=full_label + "_ctr",
        ))

    # ── forecast shading & boundary line ─────────────────────────────────────
    fig.add_vrect(
        x0=str(last_date.date()), x1=str(forecast_dates[-1].date()),
        fillcolor="rgba(255,255,255,0.04)",
        layer="below", line_width=0,
        annotation_text="◄ forecast",
        annotation_position="top left",
        annotation_font=dict(color="rgba(255,255,255,0.38)", size=11),
    )
    fig.add_vline(
        x=last_date.timestamp() * 1000,
        line=dict(color="rgba(255,255,255,0.22)", width=1, dash="dot"),
    )

    # ── MA overlays (hoverinfo="skip") ────────────────────────────────────────
    for ma_label, ma_colour, ma_vals in ma_list:
        valid = ma_vals.notna()
        fig.add_trace(go.Scatter(
            x=ma_vals.index[valid], y=ma_vals[valid],
            mode="lines", name=ma_label,
            line=dict(color=ma_colour, width=2.0),
            hoverinfo="skip",
        ))

    # ── actual price line (hoverinfo="skip") ──────────────────────────────────
    fig.add_trace(go.Scatter(
        x=hist_dates, y=prices,
        mode="lines",
        name=f"{ticker} Adj. Close",
        legendrank=10,
        line=dict(color="white", width=2),
        hoverinfo="skip",
    ))

    # ── master hover trace ────────────────────────────────────────────────────
    # Invisible line at the equilibrium centre; sole source of hover tooltips.
    # Because all other traces skip hover, this single trace fully controls the
    # popup content and ordering.
    eq_centre = next(b for b in bands if b["sigma_mult"] == 0.0)["centre"]
    fig.add_trace(go.Scatter(
        x=all_dates,
        y=eq_centre,
        mode="lines",
        line=dict(width=0, color="rgba(0,0,0,0)"),
        showlegend=False,
        hovertemplate="%{customdata}<extra></extra>",
        customdata=master_hover,
        name="_hover",
    ))

    # ── y-axis tick values and labels ─────────────────────────────────────────
    y_floor  = prices.min() * Y_FLOOR_BUFFER
    y_ceil   = prices.max() * 3.0
    log_lo   = int(np.floor(np.log10(max(y_floor, 1e-9))))
    log_hi   = int(np.ceil (np.log10(y_ceil)))
    tick_vals, tick_text = [], []
    for exp in range(log_lo, log_hi + 1):
        for mult in [1, 2, 5]:
            v = mult * 10 ** exp
            if y_floor * 0.5 <= v <= y_ceil * 2:
                tick_vals.append(v)
                if   v >= 1_000_000: tick_text.append(f"{sym}{v/1_000_000:.1f}M")
                elif v >= 1_000:     tick_text.append(f"{sym}{v/1_000:.0f}k")
                elif v >= 1:         tick_text.append(f"{sym}{v:.0f}")
                else:                tick_text.append(f"{sym}{v:.3f}")

    # ── layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        title=dict(
            text=(
                f"<b>{ticker}{ccy_note} — Rainbow Regression Chart</b><br>"
                f"<sup>Power-Law: ln(P) = {a:.4f}·ln(t) + {b:.4f}"
                f"  |  σ = {sigma:.4f}  |  {n} trading days</sup>"
            ),
            font=dict(size=18, color="white"),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title=None,
            showgrid=True, gridcolor="#1e2530",
            linecolor="#30363d",
            tickfont=dict(color="#adbac7"),
            rangeslider=dict(visible=True, thickness=0.06,
                             bgcolor="#161b22", bordercolor="#30363d"),
        ),
        yaxis=dict(
            title=f"Price (log scale, {price_currency})",
            type="log",
            showgrid=True, gridcolor="#1e2530",
            minor=dict(showgrid=True, gridcolor="#161b22", gridwidth=0.5),
            linecolor="#30363d",
            tickfont=dict(color="#adbac7"),
            tickvals=tick_vals,
            ticktext=tick_text,
            range=[np.log10(max(y_floor, 1e-9)), np.log10(y_ceil)],
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
        margin=dict(t=90, b=190, l=90, r=40),
        height=780,
    )

    # ── save ──────────────────────────────────────────────────────────────────
    save_dir.mkdir(parents=True, exist_ok=True)
    outpath = save_dir / f"{ticker}_rainbow.html"
    fig.write_html(
        str(outpath),
        include_plotlyjs="cdn",
        config={
            "scrollZoom"    : True,
            "displayModeBar": True,
            "toImageButtonOptions": {
                "format": "png", "filename": f"{ticker}_rainbow",
                "height": 900, "width": 1600, "scale": 2,
            },
        },
    )
    print(f"    💾 Interactive chart → {outpath.resolve()}")
    return fig   # returned so callers (e.g. Streamlit) can display without saving


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

def process_ticker(ticker: str, start: str, end: str) -> None:
    """End-to-end pipeline for a single ticker."""
    print(f"\n{'═' * 60}")
    print(f"  Processing  {ticker}")
    print(f"{'═' * 60}")

    # ── fetch price data ──────────────────────────────────────────────────────
    result = fetch_price_data(ticker, start, end)
    if result is None:
        return
    dates, prices = result

    # ── optional CHF conversion ───────────────────────────────────────────────
    currency       = get_native_currency(ticker)   # e.g. "USD", "EUR", "GBP"
    price_currency = currency
    if CONVERT_TO_CHF:
        rates = fetch_chf_rates(currency, dates, start, end)
        if rates is not None:
            prices         = prices * rates
            price_currency = "CHF"
        else:
            print(f"    ⚠️  CHF conversion unavailable for {currency} "
                  f"— chart will use native currency.")

    # ── fit regression ────────────────────────────────────────────────────────
    a, b, sigma, residuals = fit_log_regression(prices)

    # ── sanity-check: warn if R² is poor ──────────────────────────────────────
    n      = len(prices)
    t_hist = np.arange(1, n + 1, dtype=float)
    fitted_ln = a * np.log(t_hist) + b
    ss_res = np.sum((np.log(prices) - fitted_ln) ** 2)
    ss_tot = np.sum((np.log(prices) - np.log(prices).mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if r2 < 0.50:
        print(f"    ⚠️  Low R² ({r2:.3f}) — power-law may not fit {ticker} well in "
              f"this date range. The chart is still drawn for reference.")

    # ── plot ──────────────────────────────────────────────────────────────────
    make_rainbow_chart(ticker, dates, prices, a, b, sigma, OUTPUT_DIR, price_currency)


def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║      🌈  Stock Rainbow Regression Chart Generator  🌈      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── read settings from the top-of-file constants ────────────────────────
    tickers = [t.strip().upper() for t in TICKERS if t.strip()]
    start   = START_DATE
    end     = (datetime.today().strftime("%Y-%m-%d")
               if END_DATE.strip().lower() == "today" else END_DATE)

    print(f"\nTickers   : {', '.join(tickers)}")
    print(f"Date range: {start}  →  {end}")
    print(f"Output dir: {OUTPUT_DIR.resolve()}/\n")

    # ── process each ticker ───────────────────────────────────────────────────
    ok, failed = 0, []
    for ticker in tickers:
        try:
            process_ticker(ticker, start, end)
            ok += 1
        except Exception as exc:
            print(f"    ❌  Unexpected error for {ticker}: {exc}")
            failed.append(ticker)

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  Done.  ✅ {ok} chart(s) generated.", end="")
    if failed:
        print(f"  ❌ Failed: {', '.join(failed)}", end="")
    print(f"\n  Charts saved in: {OUTPUT_DIR.resolve()}/")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()
