"""
Alpchart — Stock Analysis Suite
"""
import warnings, os, sys, tempfile, traceback, time
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Alpchart",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Alpchart — Open-source stock analysis. No financial advice."},
)

CURRENCIES  = ["Native", "USD", "EUR", "CHF", "GBP", "JPY"]
PERIODS     = ["5Y", "10Y", "15Y", "20Y", "30Y", "50Y", "Custom"]
MAX_RAINBOW = 6
MAX_PAIRS   = 12

st.markdown("""
<style>
  /* ── Global font — scoped to text elements only, NOT icons/SVG ──── */
  body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }
  .material-icons, .material-icons-outlined, [class*="material-icon"] {
      font-family: "Material Icons", "Material Icons Outlined" !important;
  }

  /* ── Backgrounds ─────────────────────────────────────────────────── */
  .stApp { background:#0d1117; color:#e6edf3; }
  .block-container { padding-top:3.5rem !important; }

  /* ── Sidebar — uniform small font ───────────────────────────────── */
  section[data-testid="stSidebar"] { background:#161b22; }
  section[data-testid="stSidebar"] h2 {
      font-size:1.05rem !important; font-weight:600; margin-bottom:2px;
  }
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] li,
  section[data-testid="stSidebar"] small,
  section[data-testid="stSidebar"] span,
  section[data-testid="stSidebar"] a,
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] .stMarkdown,
  section[data-testid="stSidebar"] [data-testid="stMarkdown"] {
      font-size:0.79rem !important; line-height:1.4;
  }
  section[data-testid="stSidebar"] strong { font-size:0.79rem !important; }
  section[data-testid="stSidebar"] code {
      font-size:0.79rem !important; background:rgba(255,255,255,0.06);
      padding:1px 4px; border-radius:3px;
  }

  /* ── Metric cards ─────────────────────────────────────────────────── */
  div[data-testid="metric-container"] {
      background:#161b22; border:1px solid #30363d;
      border-radius:8px; padding:5px 10px;
  }
  div[data-testid="metric-container"] [data-testid="stMetricValue"] {
      font-size:0.82rem !important;
  }
  div[data-testid="metric-container"] [data-testid="stMetricLabel"] {
      font-size:0.68rem !important;
  }

  /* ── Navy buttons ─────────────────────────────────────────────────── */
  button[kind="primary"],
  [data-testid="stButton"] > button,
  div[data-testid="stButton"] > button[data-testid="baseButton-primary"] {
      background-color:#1a3a6b !important; border-color:#1a3a6b !important;
      color:#ffffff !important;
  }
  button[kind="primary"]:hover,
  [data-testid="stButton"] > button:hover,
  div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:hover {
      background-color:#0d2557 !important; border-color:#0d2557 !important;
  }

  /* ── Blue radio (period buttons) & checkbox accent ─────────────────── */
  input[type="radio"] { accent-color:#1a3a6b !important; }
  input[type="checkbox"] { accent-color:#1a3a6b !important; }

  /* ── Misc ─────────────────────────────────────────────────────────── */
  details { border:1px solid #30363d !important; border-radius:8px; }
  div[data-testid="stDownloadButton"] button {
      background:#161b22; border:1px solid #30363d; color:#79c0ff;
  }
</style>
""", unsafe_allow_html=True)


# ─────


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Alpchart")
    st.caption("Open-source stock analysis suite")
    st.divider()

    page = st.radio(
        "Tool",
        [
            "🌈 Rainbow (Power-law regression)",
            "📊 Z-Score Spread",
            "📈 P/E Ratio Spread",
        ],
        label_visibility="collapsed",
    )
    st.divider()

    st.markdown("""
**Ticker format** must match [Yahoo Finance](https://finance.yahoo.com) format:
`AAPL` · `SAP.DE` · `BTC-USD` · `^GSPC`

**Data source:** Yahoo Finance via yfinance
""")
    st.divider()
    st.markdown(
        "⭐ [Star on GitHub](https://github.com/mryqbgry27/alpchart) "
        "if you find Alpchart useful!",
    )
    st.divider()
    st.markdown("""
<small style="color:#8b949e">
⚠️ <b>Disclaimer</b><br>
Alpchart is open-source software provided "as-is" for
informational purposes only. Nothing here constitutes
financial advice. The authors accept no liability for
investment decisions made using this tool.
</small>
""", unsafe_allow_html=True)

    st.divider()
    with st.expander("Common index tickers"):
        st.markdown(
            "<style>section[data-testid='stSidebar'] table,"
            "section[data-testid='stSidebar'] td,"
            "section[data-testid='stSidebar'] th { "
            "font-size:0.72rem !important; line-height:1.3 }</style>",
            unsafe_allow_html=True)
        st.markdown("""
| Index | Ticker |
|:------|:-------|
| S&P 500 | `^GSPC` |
| Dow Jones | `^DJI` |
| Nasdaq | `^IXIC` |
| Russell 2000 | `^RUT` |
| FTSE 100 | `^FTSE` |
| CAC 40 | `^FCHI` |
| DAX | `^GDAXI` |
| Euronext 100 | `^N100` |
| SSE Composite | `000001.SS` |
| Nikkei 225 | `^N225` |
| Hang Seng | `^HSI` |
| ASX 200 | `^AXJO` |
| KOSPI | `^KS11` |
        """)
        st.caption("⚠️ No P/E data available for indices")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading modules …")
def _load_modules():
    import rainbow_regression as rr
    import zscore_spread      as zs
    import pe_ratio_spread    as pe
    return rr, zs, pe


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _subtitle(text: str) -> None:
    st.markdown(
        f"<p style='color:#8b949e;font-size:0.82rem;margin:1.1rem 0 10px 0'>{text}</p>",
        unsafe_allow_html=True,
    )

def _start_from_preset(preset: str) -> date:
    if preset == "Custom":
        return date(2012, 1, 1)
    years = int(preset[:-1])
    today = date.today()
    try:
        return today.replace(year=today.year - years)
    except ValueError:
        return today.replace(year=today.year - years, day=28)

def _parse_tickers(raw: str) -> list[str]:
    return [t.strip().upper() for t in raw.split(",") if t.strip()]

def _show_chart(html: str, height: int = 730) -> None:
    components.html(html, height=height, scrolling=False)

def _dl_btn(html: str, filename: str) -> None:
    st.download_button("💾 Download HTML", data=html.encode(),
                       file_name=filename, mime="text/html",
                       use_container_width=True)

def _chart_html(fig, inject_fn=None) -> str:
    html = fig.to_html(include_plotlyjs="cdn", config={
        "scrollZoom": True, "displayModeBar": True,
        "toImageButtonOptions": {"format": "png", "scale": 2},
    })
    return inject_fn(html) if inject_fn else html


# ─────────────────────────────────────────────────────────────────────────────
# DATE ROW  — rendered OUTSIDE forms so preset reacts immediately
# ─────────────────────────────────────────────────────────────────────────────
def _date_row(key: str) -> tuple[date, date]:
    """
    Period presets + Start/End dates. Must be placed outside st.form so the
    Custom start-date input appears reactively when 'Custom' is selected.
    """
    c_period, c_start, c_end = st.columns([5, 1.8, 1.8])

    with c_period:
        preset = st.radio(
            "Period", PERIODS, index=1, horizontal=True, key=f"preset_{key}"
        )

    is_custom = preset == "Custom"

    with c_start:
        if is_custom:
            start = st.date_input(
                "Start date",
                value=date(2012, 1, 1),
                min_value=date(1970, 1, 1),
                key=f"start_{key}",
            )
        else:
            start = _start_from_preset(preset)
            # Show a disabled widget so the layout stays stable
            st.date_input(
                "Start date",
                value=start,
                min_value=date(1970, 1, 1),
                key=f"start_{key}",
                disabled=True,
            )

    with c_end:
        end = st.date_input(
            "End date",
            value=date.today(),
            min_value=date(1970, 1, 1),
            key=f"end_{key}",
        )

    return start, end


# ─────────────────────────────────────────────────────────────────────────────
# DATA CACHING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_prices(ticker: str, start: str, end: str, which: str):
    rr, zs, pe = _load_modules()
    if which == "rainbow":
        return rr.fetch_price_data(ticker, start, end)
    fn = getattr(zs, "fetch_prices", None) or getattr(pe, "fetch_prices")
    return fn(ticker, start, end)

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_fx_raw(from_ccy: str, to_ccy: str, start: str, end: str) -> "pd.Series | None":
    if from_ccy == to_ccy:
        return None
    import yfinance as yf
    try:
        fx = yf.download(f"{from_ccy}{to_ccy}=X", start=start, end=end,
                         progress=False, auto_adjust=True)
        if fx is None or fx.empty:
            return None
        if hasattr(fx.columns, "levels"):
            fx.columns = fx.columns.get_level_values(0)
        col = "Close" if "Close" in fx.columns else fx.columns[0]
        return fx[col].dropna().astype(float)
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_eps(ticker: str, start: str, end: str) -> "pd.Series | None":
    _, _, pe = _load_modules()
    return pe.get_quarterly_eps(ticker)

def _align_rates(raw: pd.Series, price_dates: pd.DatetimeIndex) -> np.ndarray:
    return (raw.reindex(raw.index.union(price_dates))
               .ffill().reindex(price_dates).bfill().fillna(1.0).values)

def _convert(prices: np.ndarray, dates: np.ndarray,
             from_ccy: str, to_ccy: str, start: str, end: str) -> tuple[np.ndarray, str]:
    if to_ccy == "Native" or to_ccy == from_ccy:
        return prices, from_ccy
    raw = _fetch_fx_raw(from_ccy, to_ccy, start, end)
    if raw is None:
        st.warning(f"Could not fetch {from_ccy}→{to_ccy} rates; using native currency.")
        return prices, from_ccy
    pd_dates = pd.DatetimeIndex(pd.to_datetime(dates.astype("datetime64[D]")))
    return prices * _align_rates(raw, pd_dates), to_ccy

def _native_ccy(ticker: str, module) -> str:
    try:
        return module.get_native_currency(ticker)
    except Exception:
        return "USD"


# ─────────────────────────────────────────────────────────────────────────────
# ══ PAGE 1 — RAINBOW ═════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
def page_rainbow():
    rr, _, _ = _load_modules()

    _subtitle(
        "Fits a <b>power-law model</b> (ln P = a·ln t + b) to each ticker's full price history, "
        "then plots colour-coded standard-deviation bands above and below the trend. "
        "Bands reveal when a stock is historically cheap or expensive relative to its own "
        "structural growth curve — similar to the Bitcoin Rainbow Chart."
    )

    # ── Date row (outside form so Custom reacts immediately) ─────────────────
    start, end = _date_row("rr")

    # ── Config form ──────────────────────────────────────────────────────────
    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            tickers_raw = st.text_input(
                "Tickers (comma-separated)",
                "AAPL, MSFT, GOOGL, AMZN",
                help="Yahoo Finance format: AAPL · BTC-USD · SAP.DE",
            )
        with c2:
            currency = st.selectbox(
                "Display currency", CURRENCIES, index=0,
                help="Prices are converted before the regression is fitted",
            )

        with st.expander("⚙️ Chart settings"):
            a1, a2, a3, a4 = st.columns(4)
            forecast_months = a1.slider("Forecast months", 0, 36, 6)
            y_floor_buf     = a2.slider("Y-floor buffer", 0.05, 1.0, 0.20, 0.05)
            band_hw         = a3.slider("Band half-width σ", 0.1, 1.0, 0.5, 0.05)
            with a4:
                ma_200 = st.checkbox("200-day SMA", True)
                ma_600 = st.checkbox("600-day SMA", True)

    run = st.button("Generate", type="primary", use_container_width=True, key="run_rr")

    if not run:
        return

    tickers = _parse_tickers(tickers_raw)[:MAX_RAINBOW]
    if not tickers:
        st.error("Enter at least one ticker.")
        return
    if len(_parse_tickers(tickers_raw)) > MAX_RAINBOW:
        st.warning(f"Showing first {MAX_RAINBOW} tickers only.")

    rr.FORECAST_MONTHS = forecast_months
    rr.Y_FLOOR_BUFFER  = y_floor_buf
    rr.BAND_HALF_WIDTH = band_hw
    rr.MOVING_AVERAGES = ([(200, "SMA", "#00bfff")] if ma_200 else []) + \
                         ([(600, "SMA", "#ff69b4")] if ma_600 else [])

    for ticker in tickers:
        st.divider()
        st.markdown(f"**{ticker}**")
        with st.spinner(f"Fetching {ticker} …"):
            try:
                raw = _fetch_prices(ticker, str(start), str(end), "rainbow")
                if raw is None:
                    st.error(f"{ticker}: no data — check the symbol.")
                    continue
                dates, prices = raw
                native = _native_ccy(ticker, rr)
                prices, price_ccy = _convert(prices, dates, native, currency, str(start), str(end))

                a, b, sigma, _ = rr.fit_log_regression(prices)
                with tempfile.TemporaryDirectory() as tmp:
                    fig = rr.make_rainbow_chart(
                        ticker, dates, prices, a, b, sigma, Path(tmp), price_ccy
                    )

                sym = "$" if price_ccy == "USD" else f"{price_ccy} "
                st.markdown(
                    f"<div style='display:flex;flex-wrap:wrap;gap:4px 18px;"
                    f"font-size:0.74rem;background:#161b22;border:1px solid #30363d;"
                    f"border-radius:8px;padding:6px 12px;margin-bottom:4px'>"
                    f"<span><b>Latest price</b>: {sym}{prices[-1]:,.2f}</span>"
                    f"<span><b>Growth exponent</b>: {a:.4f}</span>"
                    f"<span><b>Residual \u03c3</b>: {sigma:.4f}</span>"
                    f"<span style='color:#8b949e'><b>Data points</b>: {len(prices):,}</span>"
                    f"</div>", unsafe_allow_html=True
                )

                html = _chart_html(fig)
                _show_chart(html, height=740)
                _dl_btn(html, f"{ticker}_rainbow.html")
                time.sleep(0.3)

            except Exception:
                st.error(f"{ticker}: error")
                st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# ══ PAGE 2 — Z-SCORE SPREAD ══════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
def page_zscore():
    _, zs, _ = _load_modules()

    _subtitle(
        "Normalises each asset's price against its own <b>power-law regression</b> baseline, "
        "producing a Z-score (deviation in σ units from the structural trend). "
        "The chart shows Z_A − Z_B: <b>positive = A relatively extended vs B</b>; "
        "negative = B extended vs A. SMAs of the spread reveal momentum shifts."
    )

    YF_HELP = ("Yahoo Finance format e.g. AAPL · BTC-USD · SAP.DE. "
               "Add multiple tickers with commas for matrix mode (e.g. AAPL, MSFT).")

    start, end = _date_row("zs")

    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            ra = st.text_input(
                "Ticker A — numerator(s)  *(comma-separate for multiple chart combinations)*",
                "AAPL", help=YF_HELP,
            )
            rb = st.text_input(
                "Ticker B — denominator(s)  *(comma-separate for multiple chart combinations)*",
                "MSFT", help=YF_HELP,
            )
        with c2:
            currency = st.selectbox(
                "Display currency", CURRENCIES, index=1,
                help="Both prices converted before Z-scores are computed",
                key="ccy_zs",
            )

        with st.expander("⚙️ Y-axis limits  (0 = automatic)"):
            y1, y2 = st.columns(2)
            ym = y1.number_input("Y min", 0.0, step=0.5, format="%.1f", key="ym_zs")
            yx = y2.number_input("Y max", 0.0, step=0.5, format="%.1f", key="yx_zs")
            y_min = None if ym == 0.0 else float(ym)
            y_max = None if yx == 0.0 else float(yx)

    run = st.button("Run Analysis", type="primary", use_container_width=True, key="run_zs")

    if not run:
        return

    tas = _parse_tickers(ra)
    tbs = _parse_tickers(rb)
    pairs = [(a, b) for a, b in product(tas, tbs) if a != b]
    if not pairs:
        st.error("No valid pairs — ensure Ticker A ≠ Ticker B.")
        return
    if len(pairs) > MAX_PAIRS:
        st.warning(f"Capped at {MAX_PAIRS} pairs (from {len(pairs)} combinations).")
        pairs = pairs[:MAX_PAIRS]

    zs.Y_AXIS_MIN = y_min
    zs.Y_AXIS_MAX = y_max

    for ticker_a, ticker_b in pairs:
        st.divider()
        st.markdown(f"**{ticker_a} vs {ticker_b}**")
        with st.spinner(f"Fetching {ticker_a} & {ticker_b} …"):
            try:
                pa_raw = zs.fetch_prices(ticker_a, str(start), str(end))
                pb_raw = zs.fetch_prices(ticker_b, str(start), str(end))
                if pa_raw is None or pb_raw is None:
                    st.error("Price data unavailable for one or both tickers.")
                    continue

                def _conv_series(s: pd.Series, ticker: str) -> pd.Series:
                    native = _native_ccy(ticker, zs)
                    if currency == "Native" or currency == native:
                        return s
                    raw_fx = _fetch_fx_raw(native, currency, str(start), str(end))
                    if raw_fx is None:
                        st.warning(f"{ticker}: {native}→{currency} unavailable.")
                        return s
                    return s * _align_rates(raw_fx, pd.DatetimeIndex(s.index))

                pa = _conv_series(pa_raw, ticker_a)
                pb = _conv_series(pb_raw, ticker_b)

                z_a, a_a, b_a, sig_a = zs.compute_zscore(pa, ticker_a)
                z_b, a_b, b_b, sig_b = zs.compute_zscore(pb, ticker_b)
                df = zs.compute_spread(z_a, z_b)

                spread_now = df["spread"].iloc[-1]
                sma_parts = "".join(
                    f"<span><b>{lbl.split('(')[0].strip()}</b>: "
                    f"{df[f'sma_{w}'].dropna().iloc[-1]:+.3f}\u03c3</span>"
                    for w, lbl, _ in zs.SMA_WINDOWS
                    if df[f"sma_{w}"].notna().any()
                )
                st.markdown(
                    f"<div style='display:flex;flex-wrap:wrap;gap:4px 18px;"
                    f"font-size:0.74rem;background:#161b22;border:1px solid #30363d;"
                    f"border-radius:8px;padding:6px 12px;margin-bottom:4px'>"
                    f"<span><b>Z-Spread now</b>: {spread_now:+.3f}\u03c3</span>"
                    f"{sma_parts}</div>", unsafe_allow_html=True
                )

                fig  = zs.build_chart(df, ticker_a, ticker_b,
                                      (a_a, b_a, sig_a), (a_b, b_b, sig_b))
                html = _chart_html(fig, inject_fn=zs._inject_crosshair_js)
                _show_chart(html, height=760)
                _dl_btn(html, f"{ticker_a}_vs_{ticker_b}_zscore.html")
                time.sleep(0.3)

            except Exception:
                st.error(f"{ticker_a} vs {ticker_b}: error")
                st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# ══ PAGE 3 — P/E RATIO SPREAD ════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
def page_pe():
    _, _, pe = _load_modules()

    _subtitle(
        "Compares <b>trailing twelve-month P/E ratios</b> (Price ÷ TTM EPS) for two stocks "
        "and plots how expensive A is relative to B over time. "
        "A ratio &gt; 1 means A trades at a premium; &lt; 1 means a discount. "
        "P/E is currency-neutral — exchange rates do not affect the values."
    )

    st.markdown(
        "<div style='font-size:0.79rem;color:#8b949e;background:#161b22;"
        "border:1px solid #30363d;border-radius:8px;padding:8px 12px;margin-bottom:8px'>"
        "📌 <b>EPS note:</b> Yahoo Finance provides ~4–5 years of quarterly EPS. "
        "Where only annual data is available, quarterly EPS ÷ 4 is used as a "
        "synthetic TTM — may differ slightly from official P/E figures.</div>",
        unsafe_allow_html=True)

    YF_HELP = ("Yahoo Finance format e.g. AAPL · BTC-USD · SAP.DE. "
               "Add multiple tickers with commas for matrix mode (e.g. AAPL, MSFT).")

    start, end = _date_row("pe")

    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            ra = st.text_input(
                "Ticker A — numerator(s)  *(comma-separate for multiple chart combinations)*",
                "AAPL", help=YF_HELP,
            )
            rb = st.text_input(
                "Ticker B — denominator(s)  *(comma-separate for multiple chart combinations)*",
                "MSFT", help=YF_HELP,
            )
        with c2:
            st.caption(
                "P/E is currency-neutral — exchange rates do not affect the values."
            )

        with st.expander("⚙️ Axis limits  (0 = auto)  &  EPS data"):
            r1, r2, r3, r4 = st.columns(4)
            pe_ymn = r1.number_input("P/E y-min",   0.0, step=1.0,  format="%.0f", key="pe_ymn")
            pe_ymx = r2.number_input("P/E y-max",   0.0, step=5.0,  format="%.0f", key="pe_ymx")
            r_ymn  = r3.number_input("Ratio y-min", 0.0, step=0.1,  format="%.2f", key="r_ymn")
            r_ymx  = r4.number_input("Ratio y-max", 0.0, step=0.1,  format="%.2f", key="r_ymx")
            eps_file = st.file_uploader(
                "Historical EPS CSV  *(optional — date, eps — extends beyond ~4 yrs)*",
                type=["csv"], label_visibility="visible",
            )
            if eps_file:
                st.caption("Free sources: [macrotrends.net](https://www.macrotrends.net) · "
                           "[simfin.com](https://simfin.com)")

    run = st.button("Run Analysis", type="primary", use_container_width=True, key="run_pe")

    if not run:
        return

    tas = _parse_tickers(ra)
    tbs = _parse_tickers(rb)
    pairs = [(a, b) for a, b in product(tas, tbs) if a != b]
    if not pairs:
        st.error("No valid pairs — ensure Ticker A ≠ Ticker B.")
        return
    if len(pairs) > MAX_PAIRS:
        st.warning(f"Capped at {MAX_PAIRS} pairs.")
        pairs = pairs[:MAX_PAIRS]

    pe.PE_Y_MIN    = None if pe_ymn == 0.0 else float(pe_ymn)
    pe.PE_Y_MAX    = None if pe_ymx == 0.0 else float(pe_ymx)
    pe.RATIO_Y_MIN = None if r_ymn  == 0.0 else float(r_ymn)
    pe.RATIO_Y_MAX = None if r_ymx  == 0.0 else float(r_ymx)

    tmp_csv = None
    if eps_file:
        tf = tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False)
        tf.write(eps_file.read()); tf.close()
        tmp_csv = tf.name
        pe.EPS_CSV_PATH = tmp_csv
    else:
        pe.EPS_CSV_PATH = None

    try:
        for ticker_a, ticker_b in pairs:
            st.divider()
            st.markdown(f"**{ticker_a} vs {ticker_b}**")
            with st.spinner(f"Fetching {ticker_a} & {ticker_b} …"):
                try:
                    pa = pe.fetch_prices(ticker_a, str(start), str(end))
                    pb = pe.fetch_prices(ticker_b, str(start), str(end))
                    if pa is None or pb is None:
                        st.error("Price data unavailable.")
                        continue
                    pa.name = ticker_a
                    pb.name = ticker_b

                    eps_a = _fetch_eps(ticker_a, str(start), str(end))
                    eps_b = _fetch_eps(ticker_b, str(start), str(end))
                    if eps_a is None or eps_b is None:
                        st.error("EPS data unavailable. Upload a CSV to proceed.")
                        continue

                    pea = pe.compute_ttm_pe(pa, eps_a)
                    peb = pe.compute_ttm_pe(pb, eps_b)
                    df  = pe.compute_pe_comparison(pea, peb)

                    va, vb = pea.notna().sum(), peb.notna().sum()
                    r_now  = df["ratio"].iloc[-1]
                    sma600 = (df["sma_600"].dropna().iloc[-1]
                              if df["sma_600"].notna().any() else float("nan"))

                    st.markdown(
                        f"<div style='display:flex;flex-wrap:wrap;gap:6px;font-size:0.78rem;"
                        f"background:#161b22;border:1px solid #30363d;border-radius:8px;"
                        f"padding:8px 12px;margin-bottom:6px'>"
                        f"<span style='margin-right:12px'><b>{ticker_a} P/E</b>: "
                        f"{pe._fmt_pe(pea.dropna().iloc[-1])}</span>"
                        f"<span style='margin-right:12px'><b>{ticker_b} P/E</b>: "
                        f"{pe._fmt_pe(peb.dropna().iloc[-1])}</span>"
                        f"<span style='margin-right:12px'><b>Ratio A/B</b>: "
                        f"{pe._fmt_ratio(r_now)}</span>"
                        f"<span style='margin-right:12px'><b>600-day SMA</b>: "
                        f"{pe._fmt_ratio(sma600)}</span>"
                        f"<span style='color:#8b949e'>PE days: {va} / {vb}</span></div>",
                        unsafe_allow_html=True,
                    )
                    if va < 400 or vb < 400:
                        st.caption("ℹ️ Limited EPS history — upload a CSV for full coverage.")

                    ccy_lbl = "native"
                    fig  = pe.build_chart(df, ticker_a, ticker_b, ccy_lbl)
                    html = _chart_html(fig, inject_fn=pe._inject_crosshair_js)
                    _show_chart(html, height=820)
                    _dl_btn(html, f"{ticker_a}_vs_{ticker_b}_PE.html")
                    time.sleep(0.3)

                except Exception:
                    st.error(f"{ticker_a} vs {ticker_b}: error")
                    st.code(traceback.format_exc())
    finally:
        if tmp_csv and os.path.exists(tmp_csv):
            os.unlink(tmp_csv)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────────────────────────────────────────
if page == "🌈 Rainbow (Power-law regression)":
    page_rainbow()
elif page == "📊 Z-Score Spread":
    page_zscore()
elif page == "📈 P/E Ratio Spread":
    page_pe()
