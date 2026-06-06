"""
streamlit_app.py — Stock Analysis Suite
========================================
Three tools in one app:
  🌈  Rainbow Regression Chart
  📊  Relative Z-Score Spread
  📈  Relative P/E Ratio

Run locally:   streamlit run streamlit_app.py
Deploy:        push to GitHub → connect at share.streamlit.io
"""

import warnings
import os
import sys
import tempfile
import traceback
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Analysis Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Stock Analysis Suite — Rainbow Charts · Z-Score · P/E Ratio"},
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* dark background matching the chart theme */
  .stApp            { background-color: #0d1117; color: #e6edf3; }
  section[data-testid="stSidebar"] { background-color: #161b22; }

  /* metric cards */
  div[data-testid="metric-container"] {
      background: #161b22; border: 1px solid #30363d;
      border-radius: 8px; padding: 12px 16px;
  }

  /* form border */
  div[data-testid="stForm"] {
      border: 1px solid #30363d; border-radius: 10px; padding: 18px;
  }

  /* expander */
  details { border: 1px solid #30363d !important; border-radius: 8px; }

  /* download button */
  div[data-testid="stDownloadButton"] button {
      background: #161b22; border: 1px solid #30363d; color: #79c0ff;
  }

  h1, h2, h3, h4 { color: #e6edf3; }
  .caption-text { color: #8b949e; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE LOADING  (cached so yfinance etc. are only imported once)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading modules …")
def _load_modules():
    import rainbow_regression as rr
    import zscore_spread      as zs
    import pe_ratio_spread    as pe
    return rr, zs, pe


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _show_chart(html: str, height: int = 730) -> None:
    """Embed Plotly HTML inside a Streamlit iframe."""
    components.html(html, height=height, scrolling=False)


def _download_btn(html: str, filename: str) -> None:
    st.download_button(
        label="💾 Download interactive HTML",
        data=html.encode(),
        file_name=filename,
        mime="text/html",
        use_container_width=True,
    )


def _chart_html(fig, inject_crosshair_fn=None) -> str:
    """Serialise a Plotly figure to a self-contained HTML string."""
    html = fig.to_html(
        include_plotlyjs="cdn",
        config={"scrollZoom": True, "displayModeBar": True,
                "toImageButtonOptions": {"format": "png", "scale": 2}},
    )
    if inject_crosshair_fn:
        html = inject_crosshair_fn(html)
    return html


# ─────────────────────────────────────────────────────────────────────────────
# ── DATA CACHE  (1-hour TTL avoids redundant yfinance calls)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_prices(ticker: str, start: str, end: str, module_name: str):
    rr, zs, pe = _load_modules()
    mod = {"rainbow": rr, "zscore": zs, "pe": pe}[module_name]
    fn  = getattr(mod, "fetch_price_data", None) or getattr(mod, "fetch_prices")
    return fn(ticker, start, end)


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_chf(currency: str, index_json: str, start: str, end: str, module_name: str):
    """Fetch CHF rates; index is passed as JSON string so it's hashable."""
    rr, zs, pe = _load_modules()
    mod   = {"rainbow": rr, "zscore": zs, "pe": pe}[module_name]
    index = pd.DatetimeIndex(pd.read_json(index_json, typ="series").index)
    return mod.fetch_chf_rates(currency, index, start, end)


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_eps(ticker: str, start: str, end: str):
    _, _, pe = _load_modules()
    return pe.get_quarterly_eps(ticker)


# ─────────────────────────────────────────────────────────────────────────────
# ══ PAGE 1 — RAINBOW REGRESSION CHART ════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
def page_rainbow():
    st.header("🌈 Rainbow Regression Chart")
    st.caption("Power-law regression bands reveal long-term structural valuation zones")

    with st.form("rainbow_form"):
        c1, c2, c3 = st.columns([3, 1.2, 1.2])
        with c1:
            tickers_raw = st.text_input(
                "Tickers — comma-separated",
                "AAPL, MSFT, GOOGL, AMZN",
                help="Any Yahoo Finance symbol: AAPL, BTC-USD, NESN.SW …",
            )
        with c2:
            start = st.date_input("Start date", value=date(2012, 1, 1))
            convert_chf = st.checkbox("Convert to CHF", value=True)
        with c3:
            end = st.date_input("End date", value=date.today())

        with st.expander("⚙️ Advanced settings"):
            a1, a2, a3, a4 = st.columns(4)
            forecast_months = a1.slider("Forecast months", 0, 36, 6)
            y_floor_buffer  = a2.slider("Y-floor buffer", 0.05, 1.0, 0.20, 0.05,
                                        help="Chart floor = historical_low × buffer")
            band_half_width = a3.slider("Band half-width (σ)", 0.1, 1.0, 0.5, 0.05)
            with a4:
                ma_200 = st.checkbox("200-day SMA overlay", True)
                ma_600 = st.checkbox("600-day SMA overlay", True)

        run = st.form_submit_button("🚀 Generate Charts", type="primary",
                                    use_container_width=True)

    if not run:
        return

    rr, _, _ = _load_modules()
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    if not tickers:
        st.error("Enter at least one ticker symbol.")
        return

    # Apply config
    rr.FORECAST_MONTHS   = forecast_months
    rr.Y_FLOOR_BUFFER    = y_floor_buffer
    rr.BAND_HALF_WIDTH   = band_half_width
    rr.CONVERT_TO_CHF    = convert_chf
    rr.MOVING_AVERAGES   = ([(200, "SMA", "#00bfff")] if ma_200 else []) + \
                           ([(600, "SMA", "#ff69b4")] if ma_600 else [])

    for ticker in tickers:
        st.divider()
        st.subheader(f"📈 {ticker}")

        with st.spinner(f"Fetching {ticker} …"):
            try:
                # ── fetch prices ──────────────────────────────────────────
                raw = _fetch_prices(ticker, str(start), str(end), "rainbow")
                if raw is None:
                    st.error(f"{ticker}: no price data returned — check the symbol.")
                    continue
                dates, prices = raw

                # ── optional CHF conversion ───────────────────────────────
                price_currency = rr.get_native_currency(ticker)
                if convert_chf:
                    idx_json = pd.Series(
                        index=pd.DatetimeIndex(dates.astype("datetime64[D]"))
                    ).to_json()
                    rates = _fetch_chf(price_currency, idx_json, str(start), str(end), "rainbow")
                    if rates is not None:
                        prices         = prices * rates
                        price_currency = "CHF"
                    else:
                        st.warning(f"{ticker}: CHF conversion unavailable — showing native currency.")

                # ── regression & chart ────────────────────────────────────
                a, b, sigma, _ = rr.fit_log_regression(prices)

                with tempfile.TemporaryDirectory() as tmp:
                    fig = rr.make_rainbow_chart(
                        ticker, dates, prices, a, b, sigma,
                        Path(tmp), price_currency,
                    )

                # ── summary metrics ───────────────────────────────────────
                sym = "CHF " if price_currency == "CHF" else "$"
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Latest price",    f"{sym}{prices[-1]:,.2f}")
                m2.metric("Growth exponent", f"{a:.4f}")
                m3.metric("Regression σ",    f"{sigma:.4f}")
                m4.metric("Data points",     f"{len(prices):,}")

                # ── chart ─────────────────────────────────────────────────
                html = _chart_html(fig)
                _show_chart(html, height=740)
                _download_btn(html, f"{ticker}_rainbow.html")

            except Exception:
                st.error(f"{ticker}: unexpected error.")
                st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# ══ PAGE 2 — RELATIVE Z-SCORE SPREAD ═════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
def page_zscore():
    st.header("📊 Relative Z-Score Spread")
    st.caption(
        "Each ticker's price is normalised via its own power-law regression. "
        "The spread = Z_A − Z_B reveals relative over/undervaluation."
    )

    with st.form("zscore_form"):
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.2])
        ticker_a    = c1.text_input("Ticker A  (numerator)",   "AAPL").strip().upper()
        ticker_b    = c2.text_input("Ticker B  (denominator)", "MSFT").strip().upper()
        start       = c3.date_input("Start date", value=date(2012, 1, 1))
        end         = c4.date_input("End date",   value=date.today())
        convert_chf = st.checkbox("Normalise both prices to CHF first", value=True,
                                  help="Removes USD/native-currency inflation from each series")

        with st.expander("⚙️ Y-axis limits  (leave 0 for automatic)"):
            y1, y2 = st.columns(2)
            y_min_raw = y1.number_input("Y min", value=0.0, step=0.5, format="%.1f")
            y_max_raw = y2.number_input("Y max", value=0.0, step=0.5, format="%.1f")
            y_min = None if y_min_raw == 0.0 else y_min_raw
            y_max = None if y_max_raw == 0.0 else y_max_raw

        run = st.form_submit_button("🚀 Run Analysis", type="primary",
                                    use_container_width=True)

    if not run:
        return

    if ticker_a == ticker_b:
        st.error("Ticker A and B must be different.")
        return

    _, zs, _ = _load_modules()
    zs.CONVERT_TO_CHF = convert_chf
    zs.Y_AXIS_MIN     = y_min
    zs.Y_AXIS_MAX     = y_max

    with st.spinner(f"Fetching {ticker_a} and {ticker_b} …"):
        try:
            prices_a = zs.load_ticker_prices(ticker_a, str(start), str(end))
            prices_b = zs.load_ticker_prices(ticker_b, str(start), str(end))
            if prices_a is None or prices_b is None:
                st.error("Could not fetch prices for one or both tickers.")
                return

        except Exception:
            st.error("Price fetch failed."); st.code(traceback.format_exc()); return

    with st.spinner("Computing Z-scores and spread …"):
        try:
            z_a, a_a, b_a, sig_a = zs.compute_zscore(prices_a, ticker_a)
            z_b, a_b, b_b, sig_b = zs.compute_zscore(prices_b, ticker_b)
            df = zs.compute_spread(z_a, z_b)

        except Exception:
            st.error("Computation failed."); st.code(traceback.format_exc()); return

    # ── summary metrics ───────────────────────────────────────────────────────
    spread_now = df["spread"].iloc[-1]
    cols = st.columns(1 + len(zs.SMA_WINDOWS))
    cols[0].metric("Current Z-Spread", f"{spread_now:+.3f}σ",
                   help="Positive → A extended relative to B")
    for i, (w, lbl, _) in enumerate(zs.SMA_WINDOWS, start=1):
        v = df[f"sma_{w}"].dropna().iloc[-1] if df[f"sma_{w}"].notna().any() else float("nan")
        cols[i].metric(lbl.split("(")[0].strip(), f"{v:+.3f}σ")

    # ── chart ─────────────────────────────────────────────────────────────────
    with st.spinner("Building chart …"):
        try:
            fig  = zs.build_chart(df, ticker_a, ticker_b,
                                  (a_a, b_a, sig_a), (a_b, b_b, sig_b))
            html = _chart_html(fig, inject_crosshair_fn=zs._inject_crosshair_js)
            _show_chart(html, height=760)
            _download_btn(html, f"{ticker_a}_vs_{ticker_b}_zscore.html")

        except Exception:
            st.error("Chart build failed."); st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# ══ PAGE 3 — RELATIVE P/E RATIO ══════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
def page_pe():
    st.header("📈 Relative P/E Ratio Comparison")
    st.caption(
        "Trailing TTM P/E for each stock, plus the ratio PE_A / PE_B over time. "
        "P/E is currency-neutral — CHF conversion has no effect on the values."
    )

    with st.form("pe_form"):
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.2])
        ticker_a    = c1.text_input("Ticker A  (numerator)",   "AAPL").strip().upper()
        ticker_b    = c2.text_input("Ticker B  (denominator)", "MSFT").strip().upper()
        start       = c3.date_input("Start date", value=date(2012, 1, 1))
        end         = c4.date_input("End date",   value=date.today())
        chf_label   = st.checkbox("Show [CHF] label in chart title", value=True,
                                  help="P/E itself is unaffected — this is cosmetic only")

        with st.expander("⚙️ Advanced settings"):
            r1, r2, r3, r4 = st.columns(4)
            pe_ymin_raw    = r1.number_input("P/E y-min  (0=auto)",    0.0, step=1.0,  format="%.0f")
            pe_ymax_raw    = r2.number_input("P/E y-max  (0=auto)",    0.0, step=5.0,  format="%.0f")
            ratio_ymin_raw = r3.number_input("Ratio y-min (0=auto)",   0.0, step=0.1,  format="%.2f")
            ratio_ymax_raw = r4.number_input("Ratio y-max (0=auto)",   0.0, step=0.1,  format="%.2f")

            st.markdown("**Historical EPS CSV** *(optional — extends history beyond ~4 years)*")
            eps_file = st.file_uploader(
                "Upload CSV with columns: date (YYYY-MM-DD), eps (quarterly EPS $)",
                type=["csv"], label_visibility="collapsed",
            )
            if eps_file:
                st.caption(
                    "💡 Free EPS sources: macrotrends.net · simfin.com · "
                    "financialmodelingprep.com"
                )

        run = st.form_submit_button("🚀 Run Analysis", type="primary",
                                    use_container_width=True)

    if not run:
        return

    if ticker_a == ticker_b:
        st.error("Ticker A and B must be different.")
        return

    _, _, pe = _load_modules()
    pe.CONVERT_TO_CHF = chf_label
    pe.PE_Y_MIN       = None if pe_ymin_raw    == 0.0 else pe_ymin_raw
    pe.PE_Y_MAX       = None if pe_ymax_raw    == 0.0 else pe_ymax_raw
    pe.RATIO_Y_MIN    = None if ratio_ymin_raw == 0.0 else ratio_ymin_raw
    pe.RATIO_Y_MAX    = None if ratio_ymax_raw == 0.0 else ratio_ymax_raw

    # ── optional EPS CSV ──────────────────────────────────────────────────────
    tmp_csv_path = None
    if eps_file is not None:
        tmp = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".csv", delete=False
        )
        tmp.write(eps_file.read()); tmp.close()
        tmp_csv_path = tmp.name
        pe.EPS_CSV_PATH = tmp_csv_path
    else:
        pe.EPS_CSV_PATH = None

    try:
        with st.spinner(f"Fetching prices for {ticker_a} and {ticker_b} …"):
            prices_a = _fetch_prices(ticker_a, str(start), str(end), "pe")
            prices_b = _fetch_prices(ticker_b, str(start), str(end), "pe")
            if prices_a is None or prices_b is None:
                st.error("Could not fetch prices for one or both tickers.")
                return
            prices_a.name = ticker_a
            prices_b.name = ticker_b

        with st.spinner("Fetching earnings / EPS data …"):
            eps_a = _fetch_eps(ticker_a, str(start), str(end))
            eps_b = _fetch_eps(ticker_b, str(start), str(end))
            if eps_a is None or eps_b is None:
                st.error(
                    "EPS data unavailable for one or both tickers.\n\n"
                    "Upload a historical EPS CSV above to proceed."
                )
                return

        with st.spinner("Computing TTM P/E and ratio …"):
            pea = pe.compute_ttm_pe(prices_a, eps_a)
            peb = pe.compute_ttm_pe(prices_b, eps_b)

            valid_a = pea.notna().sum()
            valid_b = peb.notna().sum()
            if valid_a < 10 or valid_b < 10:
                st.warning(
                    f"Very few valid P/E days ({valid_a} for {ticker_a}, "
                    f"{valid_b} for {ticker_b}). "
                    "Results may be unreliable — consider uploading an EPS CSV."
                )

            df = pe.compute_pe_comparison(pea, peb)

        # ── summary metrics ───────────────────────────────────────────────────
        ratio_now = df["ratio"].iloc[-1]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(f"{ticker_a} P/E",    pe._fmt_pe(pea.dropna().iloc[-1]))
        m2.metric(f"{ticker_b} P/E",    pe._fmt_pe(peb.dropna().iloc[-1]))
        m3.metric("Current ratio A/B",  pe._fmt_ratio(ratio_now))
        m4.metric("PE days (A / B)",    f"{valid_a} / {valid_b}")
        sma600 = df["sma_600"].dropna().iloc[-1] if df["sma_600"].notna().any() else float("nan")
        m5.metric("600-day SMA ratio",  pe._fmt_ratio(sma600))

        # ── EPS data quality notice ───────────────────────────────────────────
        if valid_a < 400 or valid_b < 400:
            st.info(
                "📌 **Limited EPS history** — Yahoo Finance typically provides ~4 years "
                "of quarterly earnings. Upload an EPS CSV for full historical coverage. "
                "Free source: [macrotrends.net](https://www.macrotrends.net) → "
                "search '*TICKER* EPS' → download table."
            )

        # ── chart ─────────────────────────────────────────────────────────────
        with st.spinner("Building chart …"):
            ccy_label = "CHF" if chf_label else "native"
            fig  = pe.build_chart(df, ticker_a, ticker_b, ccy_label)
            html = _chart_html(fig, inject_crosshair_fn=pe._inject_crosshair_js)
            _show_chart(html, height=810)
            _download_btn(html, f"{ticker_a}_vs_{ticker_b}_PE.html")

    except Exception:
        st.error("Unexpected error — see details below.")
        st.code(traceback.format_exc())

    finally:
        if tmp_csv_path and os.path.exists(tmp_csv_path):
            os.unlink(tmp_csv_path)


# ─────────────────────────────────────────────────────────────────────────────
# ── SIDEBAR ───────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Stock Analysis Suite")
    st.caption("Powered by yfinance + Plotly")
    st.divider()

    page = st.radio(
        "Select a tool",
        options=[
            "🌈 Rainbow Chart",
            "📊 Z-Score Spread",
            "📈 P/E Ratio",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("""
<div class='caption-text'>
<b>Data source:</b> Yahoo Finance (yfinance)<br>
<b>CHF rates:</b> Yahoo Finance forex pairs<br>
<b>P/E EPS:</b> Quarterly earnings (up to 4–5 yrs)
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── ROUTING ───────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
if page == "🌈 Rainbow Chart":
    page_rainbow()
elif page == "📊 Z-Score Spread":
    page_zscore()
elif page == "📈 P/E Ratio":
    page_pe()
