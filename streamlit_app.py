"""
Stock Analysis Suite — Streamlit App
=====================================
🌈 Rainbow Chart  |  📊 Z-Score Spread  |  📈 P/E Ratio
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

# Ensure the repo directory is on the path so the three scripts are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AlpCharts — Stock Analysis",
    page_icon="📊", layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "AlpCharts — Open-source stock analysis. No financial advice."},
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CURRENCIES   = ["Native", "USD", "EUR", "CHF", "GBP", "JPY"]
PERIODS      = ["5Y", "10Y", "15Y", "20Y", "30Y", "50Y", "Custom"]
MAX_RAINBOW  = 6    # max individual tickers on rainbow page
MAX_PAIRS    = 12   # hard cap on matrix combinations (z-score & PE)

# ─────────────────────────────────────────────────────────────────────────────
# CSS  — compact, dark, GitHub-style
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp                          { background:#0d1117; color:#e6edf3; }
  section[data-testid="stSidebar"]{ background:#161b22; }
  div[data-testid="stForm"]       { border:1px solid #30363d; border-radius:10px; padding:14px 16px 10px; }
  div[data-testid="metric-container"]{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:8px 12px; }
  /* tighten default page top-padding */
  .block-container { padding-top:0.6rem !important; }
  /* smaller metric numbers */
  div[data-testid="metric-container"] [data-testid="stMetricValue"]{ font-size:0.95rem !important; }
  div[data-testid="metric-container"] [data-testid="stMetricLabel"]{ font-size:0.7rem !important; }
  h1,h2,h3,h4 { color:#e6edf3; margin-bottom:2px !important; }
  p  { margin-top:2px !important; }
  details { border:1px solid #30363d !important; border-radius:8px; }
  div[data-testid="stDownloadButton"] button { background:#161b22; border:1px solid #30363d; color:#79c0ff; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 AlpCharts")
    st.caption("Open-source stock analysis suite")
    st.divider()

    page = st.radio("Tool", ["🌈 Rainbow", "📊 Z-Score Spread", "📈 P/E Ratio"],
                    label_visibility="collapsed")
    st.divider()

    st.markdown("""
**Ticker format**
Symbols must match [Yahoo Finance](https://finance.yahoo.com) format:
`AAPL` · `NESN.SW` · `BTC-USD` · `^GSPC`

**Data source:** Yahoo Finance via `yfinance`
""")
    st.divider()

    st.markdown("""
<small style="color:#8b949e">
⚠️ <b>Disclaimer</b><br>
AlpCharts is open-source software provided "as-is" for
informational purposes only. Nothing here constitutes
financial advice. The authors accept no liability for
investment decisions made using this tool.
</small>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE LOADING  (once per process)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading analysis modules …")
def _load_modules():
    import rainbow_regression as rr
    import zscore_spread      as zs
    import pe_ratio_spread    as pe
    return rr, zs, pe


# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def _hdr(title: str, sub: str = "") -> None:
    """Compact page header."""
    st.markdown(f"<h3 style='margin:0 0 1px 0'>{title}</h3>", unsafe_allow_html=True)
    if sub:
        st.markdown(f"<p style='color:#8b949e;font-size:0.82rem;margin:0 0 8px 0'>{sub}</p>",
                    unsafe_allow_html=True)


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
# DATA CACHING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_prices(ticker: str, start: str, end: str, which: str) -> "tuple | None":
    """Return raw price data from the appropriate script."""
    rr, zs, pe = _load_modules()
    if which == "rainbow":
        return rr.fetch_price_data(ticker, start, end)   # returns (dates, prices)
    fn = getattr(zs, "fetch_prices", None) or getattr(pe, "fetch_prices")
    s  = fn(ticker, start, end)
    return (None, s) if s is None else s   # returns pd.Series directly


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_fx_raw(from_ccy: str, to_ccy: str, start: str, end: str) -> "pd.Series | None":
    """Fetch raw FX rates as a date-indexed Series; returns None if same currency."""
    if from_ccy == to_ccy:
        return None
    import yfinance as yf
    fx_ticker = f"{from_ccy}{to_ccy}=X"
    try:
        fx = yf.download(fx_ticker, start=start, end=end, progress=False, auto_adjust=True)
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
             from_ccy: str, to_ccy: str,
             start: str, end: str) -> tuple[np.ndarray, str]:
    """Convert a price array from from_ccy to to_ccy; returns (prices, actual_ccy)."""
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
# DATE PRESET WIDGET
# ─────────────────────────────────────────────────────────────────────────────
def _date_widgets(key: str) -> tuple[date, date]:
    """Render period preset + optional custom start, return (start, end)."""
    ca, cb = st.columns([3, 1])
    with ca:
        preset = st.radio("Period", PERIODS, index=1, horizontal=True, key=f"preset_{key}")
    with cb:
        end = st.date_input("End", value=date.today(), min_value=date(1970, 1, 1), key=f"end_{key}")
    if preset == "Custom":
        start = st.date_input("Start date", value=date(2012, 1, 1),
                              min_value=date(1970, 1, 1), key=f"start_{key}")
    else:
        start = _start_from_preset(preset)
        st.caption(f"📅 Start: **{start}**")
    return start, end


# ─────────────────────────────────────────────────────────────────────────────
# ══ PAGE 1 — RAINBOW CHART ═══════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
def page_rainbow():
    rr, _, _ = _load_modules()
    _hdr("🌈 Rainbow Regression Chart",
         "Power-law regression bands — structural valuation zones per ticker")

    with st.form("rainbow_form"):
        c1, c2 = st.columns([3, 1])
        with c1:
            tickers_raw = st.text_input(
                "Tickers (comma-separated)", "AAPL, MSFT, GOOGL, AMZN",
                help="Yahoo Finance format: AAPL · BTC-USD · NESN.SW")
            currency = st.selectbox("Display currency", CURRENCIES, index=3,
                                    help="Prices converted before regression is fitted")
        with c2:
            start, end = _date_widgets("rr")

        with st.expander("⚙️ Chart settings"):
            a1, a2, a3, a4 = st.columns(4)
            forecast_months = a1.slider("Forecast months", 0, 36, 6)
            y_floor_buf     = a2.slider("Y-floor buffer",  0.05, 1.0, 0.20, 0.05)
            band_hw         = a3.slider("Band half-width σ", 0.1, 1.0, 0.5, 0.05)
            with a4:
                ma_200 = st.checkbox("200-day SMA", True)
                ma_600 = st.checkbox("600-day SMA", True)

        run = st.form_submit_button("🚀 Generate", type="primary", use_container_width=True)

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
                    fig = rr.make_rainbow_chart(ticker, dates, prices, a, b, sigma,
                                                Path(tmp), price_ccy)

                sym = f"{price_ccy} " if price_ccy not in ("USD",) else "$"
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Latest price",    f"{sym}{prices[-1]:,.2f}")
                m2.metric("Growth exponent", f"{a:.4f}")
                m3.metric("Residual σ",      f"{sigma:.4f}")
                m4.metric("Data points",     f"{len(prices):,}")

                _show_chart(_chart_html(fig), height=740)
                _dl_btn(_chart_html(fig), f"{ticker}_rainbow.html")
                time.sleep(0.3)   # gentle rate-limit between tickers

            except Exception:
                st.error(f"{ticker}: error"); st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# ══ PAGE 2 — Z-SCORE SPREAD ══════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
def page_zscore():
    _, zs, _ = _load_modules()
    _hdr("📊 Relative Z-Score Spread",
         "Power-law Z-score of A minus B — positive = A relatively extended vs B")

    with st.form("zscore_form"):
        c1, c2 = st.columns([3, 1])
        with c1:
            YF_HELP = "Yahoo Finance format e.g. AAPL, MSFT, BTC-USD  (comma-separate for matrix mode)"
            ra = st.text_input("Ticker A — numerator(s)",   "AAPL", help=YF_HELP)
            rb = st.text_input("Ticker B — denominator(s)", "MSFT", help=YF_HELP)
            currency = st.selectbox("Display currency", CURRENCIES, index=3,
                                    help="Both prices converted before Z-score is computed")
        with c2:
            start, end = _date_widgets("zs")

        with st.expander("⚙️ Y-axis limits  (0 = automatic)"):
            y1, y2 = st.columns(2)
            ym = y1.number_input("Y min", 0.0, step=0.5, format="%.1f")
            yx = y2.number_input("Y max", 0.0, step=0.5, format="%.1f")
            y_min = None if ym == 0.0 else float(ym)
            y_max = None if yx == 0.0 else float(yx)

        run = st.form_submit_button("🚀 Run Analysis", type="primary", use_container_width=True)

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
                # Fetch raw prices (Series)
                pa_raw = zs.fetch_prices(ticker_a, str(start), str(end))
                pb_raw = zs.fetch_prices(ticker_b, str(start), str(end))
                if pa_raw is None or pb_raw is None:
                    st.error(f"Price data unavailable for one or both tickers.")
                    continue

                # Currency conversion using dates from the Series index
                def _conv_series(s: pd.Series, ticker: str) -> pd.Series:
                    native = _native_ccy(ticker, zs)
                    if currency == "Native" or currency == native:
                        return s
                    raw_fx = _fetch_fx_raw(native, currency, str(start), str(end))
                    if raw_fx is None:
                        st.warning(f"{ticker}: {native}→{currency} unavailable.")
                        return s
                    aligned = _align_rates(raw_fx, pd.DatetimeIndex(s.index))
                    return s * aligned

                pa = _conv_series(pa_raw, ticker_a)
                pb = _conv_series(pb_raw, ticker_b)

                z_a, a_a, b_a, sig_a = zs.compute_zscore(pa, ticker_a)
                z_b, a_b, b_b, sig_b = zs.compute_zscore(pb, ticker_b)
                df = zs.compute_spread(z_a, z_b)

                spread_now = df["spread"].iloc[-1]
                cols = st.columns(1 + len(zs.SMA_WINDOWS))
                cols[0].metric("Z-Spread now", f"{spread_now:+.3f}σ")
                for i, (w, lbl, _) in enumerate(zs.SMA_WINDOWS, 1):
                    v = df[f"sma_{w}"].dropna().iloc[-1] if df[f"sma_{w}"].notna().any() else float("nan")
                    cols[i].metric(lbl.split("(")[0].strip(), f"{v:+.3f}σ")

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
# ══ PAGE 3 — P/E RATIO ═══════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
def page_pe():
    _, _, pe = _load_modules()
    _hdr("📈 Relative P/E Ratio",
         "Trailing TTM P/E per stock + ratio over time  — P/E is currency-neutral")

    st.info(
        "📌 **EPS note:** Yahoo Finance provides ~4–5 years of quarterly EPS. "
        "Where only annual data is available the quarterly value is estimated as "
        "annual EPS ÷ 4, producing a **synthetic TTM**. This may differ slightly "
        "from official trailing P/E numbers.",
        icon=None,
    )

    with st.form("pe_form"):
        c1, c2 = st.columns([3, 1])
        with c1:
            YF_HELP = "Yahoo Finance format e.g. AAPL, MSFT  (comma-separate for matrix mode)"
            ra = st.text_input("Ticker A — numerator(s)",   "AAPL", help=YF_HELP)
            rb = st.text_input("Ticker B — denominator(s)", "MSFT", help=YF_HELP)
            chf_label = st.checkbox("Show [CHF] label", True,
                                    help="P/E is currency-neutral; this is a cosmetic label only")
        with c2:
            start, end = _date_widgets("pe")

        with st.expander("⚙️ Axis limits  (0 = auto)  &  EPS data"):
            r1, r2, r3, r4 = st.columns(4)
            pe_ymn = r1.number_input("P/E y-min",    0.0, step=1.0,  format="%.0f")
            pe_ymx = r2.number_input("P/E y-max",    0.0, step=5.0,  format="%.0f")
            r_ymn  = r3.number_input("Ratio y-min",  0.0, step=0.1,  format="%.2f")
            r_ymx  = r4.number_input("Ratio y-max",  0.0, step=0.1,  format="%.2f")
            eps_file = st.file_uploader(
                "Historical EPS CSV (optional — date, eps columns — extends history beyond ~4 yrs)",
                type=["csv"], label_visibility="visible",
            )

        run = st.form_submit_button("🚀 Run Analysis", type="primary", use_container_width=True)

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

    # optional CSV
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
                        st.error("Price data unavailable."); continue
                    pa.name = ticker_a; pb.name = ticker_b

                    eps_a = _fetch_eps(ticker_a, str(start), str(end))
                    eps_b = _fetch_eps(ticker_b, str(start), str(end))
                    if eps_a is None or eps_b is None:
                        st.error("EPS data unavailable. Upload a CSV to proceed."); continue

                    pea = pe.compute_ttm_pe(pa, eps_a)
                    peb = pe.compute_ttm_pe(pb, eps_b)
                    df  = pe.compute_pe_comparison(pea, peb)

                    # ── compact metrics ───────────────────────────────────
                    va, vb = pea.notna().sum(), peb.notna().sum()
                    r_now  = df["ratio"].iloc[-1]
                    sma600 = df["sma_600"].dropna().iloc[-1] if df["sma_600"].notna().any() else float("nan")
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
                        st.caption("ℹ️ Limited EPS history — upload a CSV for full coverage. "
                                   "Free source: [macrotrends.net](https://www.macrotrends.net)")

                    ccy_lbl = "CHF" if chf_label else "native"
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
if page == "🌈 Rainbow":
    page_rainbow()
elif page == "📊 Z-Score Spread":
    page_zscore()
elif page == "📈 P/E Ratio":
    page_pe()
