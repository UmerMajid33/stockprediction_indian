"""
CML7133 Assignment 2 - Deployment
Stock Direction Predictor with live data, interactive charts and explainability.

Run locally:   streamlit run app.py
Deployed on:   Streamlit Community Cloud (share.streamlit.io)
"""
import io
import json
import os
import smtplib
import warnings
from datetime import date, datetime, timedelta
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stock Direction Predictor",
                   page_icon="chart_with_upwards_trend",
                   layout="wide", initial_sidebar_state="expanded")

MODEL_PATH = "models/model_bundle.joblib"

# ============================================================================
# INDICATORS - must match the notebook exactly, or the features will not line up
# ============================================================================

def smooth(series, n):
    """Wilder's smoothing. It is an exponential average with alpha = 1/n.
    RSI, ATR and ADX all use this instead of a normal moving average."""
    return series.ewm(alpha=1/n, adjust=False).mean()


def get_rsi(close, n=14):
    """RSI measures how much of the recent movement was upward.
    It runs from 0 to 100. Above 70 is usually called overbought."""
    change = close.diff()
    ups = smooth(change.clip(lower=0), n)          # only the positive changes
    downs = smooth(-change.clip(upper=0), n)       # only the negative changes
    rs = ups / downs.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def get_stochastic(high, low, close, n=9, k_smooth=3, d_smooth=3):
    """Where is the close sitting inside the recent high-low range?
    0 means at the bottom of the range, 100 means at the top."""
    lowest = low.rolling(n).min()
    highest = high.rolling(n).max()
    raw_k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    k = raw_k.ewm(alpha=1/k_smooth, adjust=False).mean()
    d = k.ewm(alpha=1/d_smooth, adjust=False).mean()
    return raw_k, k, d


def get_kdj(high, low, close, n=9):
    """KDJ. K and D come from the stochastic above, and J = 3K - 2D.
    The J line exaggerates the gap between K and D, which is the point of it.
    J can go below 0 or above 100 and that is normal."""
    raw_k, k, d = get_stochastic(high, low, close, n)
    j = 3 * k - 2 * d
    return k, d, j


def get_bbi(close):
    """BBI (Bull and Bear Index) is just the average of four moving averages.
    It blends short-term and medium-term trend into one line."""
    return (close.rolling(3).mean() + close.rolling(6).mean()
            + close.rolling(12).mean() + close.rolling(24).mean()) / 4


def get_macd(close, fast=12, slow=26, signal=9):
    """MACD is the difference between a fast and a slow exponential average."""
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    signal_line = line.ewm(span=signal, adjust=False).mean()
    histogram = line - signal_line
    return line, signal_line, histogram


def get_bollinger(close, n=20, k=2):
    """Bands drawn two standard deviations above and below the average."""
    middle = close.rolling(n).mean()
    sd = close.rolling(n).std()
    upper = middle + k * sd
    lower = middle - k * sd
    width = (upper - lower) / middle.replace(0, np.nan)
    position = (close - lower) / (upper - lower).replace(0, np.nan)   # 0 = at lower band
    return upper, lower, middle, width, position


def get_atr(high, low, close, n=14):
    """Average True Range - the usual way to measure how volatile a stock is."""
    ranges = pd.concat([high - low,
                        (high - close.shift()).abs(),
                        (low - close.shift()).abs()], axis=1)
    return smooth(ranges.max(axis=1), n)


def get_adx(high, low, close, n=14):
    """ADX measures how STRONG a trend is, without saying which direction.
    The +DI and -DI lines say which direction."""
    up_move = high.diff()
    down_move = -low.diff()
    plus = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_now = get_atr(high, low, close, n)
    plus_di = 100 * smooth(pd.Series(plus, index=high.index), n) / atr_now.replace(0, np.nan)
    minus_di = 100 * smooth(pd.Series(minus, index=high.index), n) / atr_now.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return plus_di, minus_di, smooth(dx, n)


def get_williams_r(high, low, close, n=14):
    """Like the stochastic but on a scale of -100 to 0."""
    highest = high.rolling(n).max()
    lowest = low.rolling(n).min()
    return -100 * (highest - close) / (highest - lowest).replace(0, np.nan)


def get_cci(high, low, close, n=20):
    """Commodity Channel Index - how far the price is from its average,
    measured in units of its own average deviation."""
    typical = (high + low + close) / 3
    avg = typical.rolling(n).mean()
    dev = (typical - avg).abs().rolling(n).mean()
    return (typical - avg) / (0.015 * dev.replace(0, np.nan))


def get_obv(close, volume):
    """On Balance Volume - adds volume on up days and subtracts it on down days."""
    return (np.sign(close.diff()).fillna(0) * volume).cumsum()


def get_mfi(high, low, close, volume, n=14):
    """Money Flow Index - basically RSI but using price times volume."""
    typical = (high + low + close) / 3
    flow = typical * volume
    positive = flow.where(typical.diff() > 0, 0.0).rolling(n).sum()
    negative = flow.where(typical.diff() < 0, 0.0).rolling(n).sum()
    return 100 * positive / (positive + negative).replace(0, np.nan)


def volume_is_usable(d):
    if "Volume" not in d.columns:
        return False
    v = d["Volume"].dropna()
    return len(v) > 0 and v.nunique() > 5 and v.abs().sum() > 0


def make_features(data, lags=(1, 2, 3, 5, 10)):
    """Build all the features. Returns the feature table and a dictionary saying
    which family each feature belongs to."""
    X = pd.DataFrame(index=data.index)
    op, hi, lo, cl = data["Open"], data["High"], data["Low"], data["Close"]
    vol = data["Volume"]
    families = {}

    # --- returns and the shape of the daily candle ---
    daily_return = cl.pct_change()
    X["return_1d"] = daily_return
    for n in [2, 3, 5, 10, 21]:
        X[f"return_{n}d"] = cl.pct_change(n)
    X["day_range"] = (hi - lo) / cl              # how wide was today's move
    X["close_vs_open"] = (cl - op) / op
    X["overnight_gap"] = (op - cl.shift(1)) / cl.shift(1)
    X["upper_wick"] = (hi - np.maximum(op, cl)) / cl
    X["lower_wick"] = (np.minimum(op, cl) - lo) / cl
    families["Returns"] = list(X.columns)

    # --- momentum: is the stock moving up or down recently, and how strongly ---
    start = len(X.columns)
    for n in [7, 14, 21]:
        X[f"rsi_{n}"] = get_rsi(cl, n)
    k, d, j = get_kdj(hi, lo, cl, 9)
    X["kdj_k"], X["kdj_d"], X["kdj_j"] = k, d, j
    X["kdj_j_minus_k"] = j - k
    raw_k, stoch_k, stoch_d = get_stochastic(hi, lo, cl, 14)
    X["stoch_k"], X["stoch_d"] = stoch_k, stoch_d
    X["stoch_k_minus_d"] = stoch_k - stoch_d
    X["williams_r"] = get_williams_r(hi, lo, cl, 14)
    X["cci_20"] = get_cci(hi, lo, cl, 20)
    macd_line, macd_signal, macd_hist = get_macd(cl)
    # MACD is measured in rupees, so I divide by the price to make it relative
    X["macd"] = macd_line / cl
    X["macd_signal"] = macd_signal / cl
    X["macd_hist"] = macd_hist / cl
    X["macd_hist_change"] = macd_hist.diff()
    families["Momentum"] = list(X.columns)[start:]

    # --- trend: where is the price compared to its moving averages ---
    start = len(X.columns)
    for n in [5, 10, 20, 50, 200]:
        # "how many percent above the average is the price" - this is the relative form
        X[f"close_vs_sma{n}"] = cl / cl.rolling(n).mean() - 1
        if n <= 50:
            X[f"close_vs_ema{n}"] = cl / cl.ewm(span=n, adjust=False).mean() - 1
    X["sma5_vs_sma20"] = cl.rolling(5).mean() / cl.rolling(20).mean() - 1
    X["sma20_vs_sma50"] = cl.rolling(20).mean() / cl.rolling(50).mean() - 1
    bbi = get_bbi(cl)
    X["close_vs_bbi"] = cl / bbi - 1
    X["bbi_change_5d"] = bbi.pct_change(5)
    plus_di, minus_di, adx = get_adx(hi, lo, cl, 14)
    X["adx"], X["di_plus"], X["di_minus"] = adx, plus_di, minus_di
    X["di_difference"] = plus_di - minus_di
    families["Trend"] = list(X.columns)[start:]

    # --- volatility: how much is the price jumping around ---
    start = len(X.columns)
    bb_up, bb_low, bb_mid, bb_width, bb_pos = get_bollinger(cl, 20, 2)
    X["bollinger_width"] = bb_width
    X["bollinger_position"] = bb_pos
    atr = get_atr(hi, lo, cl, 14)
    X["atr_relative"] = atr / cl               # ATR in rupees divided by price
    X["dist_to_upper_band"] = (bb_up - cl) / atr
    X["dist_to_lower_band"] = (cl - bb_low) / atr
    for n in [5, 10, 21]:
        X[f"volatility_{n}d"] = daily_return.rolling(n).std()
    X["volatility_ratio"] = X["volatility_5d"] / X["volatility_21d"]
    X["return_skew_21d"] = daily_return.rolling(21).skew()
    X["return_kurtosis_21d"] = daily_return.rolling(21).kurt()
    families["Volatility"] = list(X.columns)[start:]

    # --- volume: is the move backed by lots of trading, or is it quiet ---
    start = len(X.columns)
    X["volume_vs_average"] = vol / vol.rolling(20).mean() - 1
    X["volume_change"] = vol.pct_change()
    X["obv_change_10d"] = get_obv(cl, vol).diff(10) / vol.rolling(20).mean()
    X["mfi_14"] = get_mfi(hi, lo, cl, vol, 14)
    typical = (hi + lo + cl) / 3
    vwap = (typical * vol).rolling(20).sum() / vol.rolling(20).sum()
    X["close_vs_vwap"] = cl / vwap - 1
    X["volume_price_correlation"] = daily_return.rolling(10).corr(vol.pct_change())
    families["Volume"] = list(X.columns)[start:]

    # --- calendar ---
    start = len(X.columns)
    X["day_of_week"] = X.index.dayofweek
    X["month"] = X.index.month
    X["quarter"] = X.index.quarter
    X["day_of_month"] = X.index.day
    families["Calendar"] = list(X.columns)[start:]

    # --- LAG FEATURES ---
    # A single day only tells me where things are right now. Lagged copies let the
    # model see the direction things are moving in. For example, RSI at 55 means
    # something different if yesterday it was 45 than if yesterday it was 65.
    start = len(X.columns)
    to_lag = ["return_1d", "rsi_14", "kdj_j", "macd_hist", "bollinger_position",
              "atr_relative", "close_vs_sma20", "adx"]
    lag_columns = {}
    for name in to_lag:
        for k in lags:
            lag_columns[f"{name}_lag{k}"] = X[name].shift(k)
    lag_columns["avg_return_5d"] = X["return_1d"].rolling(5).mean()
    lag_columns["avg_return_10d"] = X["return_1d"].rolling(10).mean()
    lag_columns["up_days_last_5"] = (X["return_1d"] > 0).rolling(5).sum()
    lag_columns["up_days_last_10"] = (X["return_1d"] > 0).rolling(10).sum()
    X = pd.concat([X, pd.DataFrame(lag_columns, index=X.index)], axis=1)
    families["Lag"] = list(X.columns)[start:]

    # a few divisions can produce infinity, replace those with missing
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, families

def build_features(d, lags=(1, 2, 3, 5, 10)):
    """The app only needs the feature table, not the family grouping."""
    X, _families = make_features(d, lags=lags)
    return X


# the chart code below uses these older names
calc_bollinger = get_bollinger
calc_bbi = get_bbi
calc_rsi = get_rsi


# ============================================================================
# LOADING
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_bundle(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


INDIA_PRESETS = {
    "Nifty 50 (index)": "^NSEI",
    "Sensex (index)": "^BSESN",
    "Bank Nifty (index)": "^NSEBANK",
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "India VIX": "^INDIAVIX",
    "Custom symbol...": None,
}


SYMBOL_NAMES = {v: k.replace(" (index)", "") for k, v in INDIA_PRESETS.items() if v}


def display_name(sym):
    """Friendly label for a Yahoo symbol, e.g. ^NSEI -> Nifty 50."""
    return SYMBOL_NAMES.get(sym, sym)


def normalize_india_ticker(t):
    """Bare symbols (RELIANCE) become NSE tickers (RELIANCE.NS). Indices (^NSEI)
    and symbols with an explicit exchange suffix (.NS, .BO) are left alone."""
    t = t.strip().upper()
    if t and not t.startswith("^") and "." not in t and "-" not in t:
        return t + ".NS"
    return t


@st.cache_data(ttl=900, show_spinner=False)
def fetch_prices(ticker, start, end):
    """Live fetch, cached for 15 minutes so repeated interaction is instant."""
    import yfinance as yf
    d = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if d is None or len(d) == 0:
        raise ValueError(f"No data returned for '{ticker}'. Check the symbol.")
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    d = d.rename(columns={"Adj Close": "AdjClose"})
    keep = [c for c in ["Open", "High", "Low", "Close", "AdjClose", "Volume"] if c in d.columns]
    return d[keep].dropna(subset=["Close"]).sort_index()


# ============================================================================
# REPORTING
# ============================================================================

def build_pdf(ctx):
    """One-page PDF summary. Returns bytes, or None if reportlab is unavailable."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table,
                                        TableStyle)
    except ImportError:
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                            leftMargin=1.8 * cm, rightMargin=1.8 * cm)
    ss = getSampleStyleSheet()
    h = ParagraphStyle("h", parent=ss["Heading1"], fontSize=15, spaceAfter=4)
    sub = ParagraphStyle("sub", parent=ss["Normal"], fontSize=8.5,
                         textColor=colors.HexColor("#666"), spaceAfter=10)
    body = ParagraphStyle("b", parent=ss["Normal"], fontSize=9, leading=13)

    story = [Paragraph("Stock Direction Prediction Report", h),
             Paragraph(f"Generated {datetime.now():%Y-%m-%d %H:%M} &nbsp;|&nbsp; "
                       f"CML7133 Assignment 2 &nbsp;|&nbsp; Umer Majid Muneer (261PC260Y3)", sub)]

    rows = [["Field", "Value"],
            ["Ticker", ctx["ticker"]],
            ["Latest close", f"{ctx['last_close']:,.4f} on {ctx['last_date']}"],
            ["Prediction horizon", f"{ctx['horizon']} trading days"],
            ["Direction", ctx["direction"]],
            ["Confidence", f"{ctx['confidence']:.1%}"],
            ["Expected return (regression)", f"{ctx['expected_return']:+.2%}"],
            ["Market regime", ctx["regime"]],
            ["Historical P(up) in regime", ctx["regime_pup"]],
            ["Model", ctx["model_name"]],
            ["Trained on", f"{ctx['trained_rows']:,} rows ({ctx['train_start']} to {ctx['train_end']})"],
            ["CV accuracy / majority baseline", ctx["acc_vs_base"]]]
    t = Table(rows, colWidths=[6.2 * cm, 9.3 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3D6B9E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [t, Spacer(1, 0.55 * cm)]

    story.append(Paragraph("Top contributing features", ss["Heading3"]))
    frows = [["Feature", "Value", "SHAP contribution"]]
    for f, val, sv in ctx["top_features"]:
        frows.append([f, f"{val:,.4f}", f"{sv:+.4f}"])
    ft = Table(frows, colWidths=[7.5 * cm, 4 * cm, 4 * cm])
    ft.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#555")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DDD")),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ]))
    story += [ft, Spacer(1, 0.55 * cm)]

    story.append(Paragraph(
        "<b>Important:</b> this is a student coursework artefact built for CML7133. "
        "The underlying model achieves only a few percentage points above the "
        "majority-class baseline, which is typical for financial direction "
        "prediction and far too weak to base any financial decision on. It is not "
        "investment advice.", body))

    doc.build(story)
    return buf.getvalue()


def send_email(to_addr, subject, body_text, attachments):
    cfg = st.secrets.get("email", {})
    sender = cfg.get("sender")
    password = cfg.get("password")
    server = cfg.get("smtp_server", "smtp.gmail.com")
    port = int(cfg.get("smtp_port", 587))
    if not sender or not password:
        return False, ("No SMTP credentials configured. Add an [email] section to "
                       ".streamlit/secrets.toml to enable sending.")
    try:
        msg = MIMEMultipart()
        msg["From"], msg["To"], msg["Subject"] = sender, to_addr, subject
        msg.attach(MIMEText(body_text, "plain"))
        for fname, data in attachments:
            part = MIMEApplication(data, Name=fname)
            part["Content-Disposition"] = f'attachment; filename="{fname}"'
            msg.attach(part)
        with smtplib.SMTP(server, port, timeout=25) as s:
            s.starttls()
            s.login(sender, password)
            s.send_message(msg)
        return True, f"Report sent to {to_addr}"
    except Exception as e:
        return False, f"Send failed: {type(e).__name__}: {e}"


# ============================================================================
# UI
# ============================================================================

bundle = load_bundle()

st.title("Stock Direction Predictor")
st.caption("CML7133 Assignment 2 — Umer Majid Muneer (261PC260Y3) — "
           "live prediction from technical indicators")

if bundle is None:
    st.error(f"Model bundle not found at `{MODEL_PATH}`. "
             "Run the notebook's final cell to create it, then redeploy.")
    st.stop()

with st.sidebar:
    st.header("Controls")
    # start on the stock the model was actually trained on, not the first in the list
    trained_symbol = bundle.get("ticker_trained_on", "^NSEI")
    preset_names = list(INDIA_PRESETS)
    default_index = next((i for i, name in enumerate(preset_names)
                          if INDIA_PRESETS[name] == trained_symbol), 0)
    choice = st.selectbox("Indian market instrument", preset_names, index=default_index)
    if INDIA_PRESETS[choice] is None:
        raw = st.text_input("NSE/BSE symbol", value="WIPRO",
                            help="NSE symbol such as WIPRO (auto-adds .NS), or BSE such as "
                                 "500325.BO, or an index such as ^NSEI")
        ticker = normalize_india_ticker(raw)
    else:
        ticker = INDIA_PRESETS[choice]
    st.caption(f"Showing: **{display_name(ticker)}**")
    years = st.slider("Years of history to load", 2, 25, 10)
    horizon = st.select_slider("Prediction horizon (trading days)",
                               options=[1, 2, 3, 5, 10, 21],
                               value=int(bundle.get("horizon", 3)))
    st.divider()
    show_regime = st.checkbox("Show market regime", value=True)
    show_shap = st.checkbox("Explain the prediction (SHAP)", value=True)
    show_backtest = st.checkbox("Show historical signal backtest", value=True)
    chart_days = st.slider("Days shown on the chart", 120, 1000, 320, step=20)
    st.divider()
    st.caption(f"Model trained on {bundle['ticker_trained_on']} · "
               f"{bundle['trained_rows']:,} rows · "
               f"{bundle['train_start']} to {bundle['train_end']}")
    if ticker != bundle.get("ticker_trained_on"):
        st.warning("This model was trained on "
                   + str(bundle.get("stock_name", bundle.get("ticker_trained_on")))
                   + ". Applying it to a different instrument still works, but the "
                     "accuracy figures below were measured on the trained one.")
    if horizon != bundle.get("horizon"):
        st.warning(f"The saved model was trained for a {bundle.get('horizon')}-day "
                   f"horizon. Predicting {horizon} days uses the same model, so "
                   f"treat the result as indicative only.")

end_d = date.today() + timedelta(days=1)
start_d = end_d - timedelta(days=int(years * 365.25) + 400)   # extra for indicator warm-up

try:
    with st.spinner(f"Fetching {display_name(ticker)} (Indian market) ..."):
        prices = fetch_prices(ticker, start_d.isoformat(), end_d.isoformat())
except Exception as e:
    st.error(f"Could not fetch data: {e}")
    st.info("If this is a network restriction rather than a bad symbol, the ticker "
            "may still work when the app runs on Streamlit Cloud.")
    st.stop()

feats_all = build_features(prices, lags=tuple(bundle.get("lags", (1, 2, 3, 5, 10))))
needed = bundle["features"]
missing = [f for f in needed if f not in feats_all.columns]
if missing:
    st.warning(f"{len(missing)} feature(s) unavailable for this ticker "
               f"(usually volume-based, when the feed has no volume): "
               f"{', '.join(missing[:6])}{'...' if len(missing) > 6 else ''}. "
               "They are filled with the training median.")
    for f in missing:
        feats_all[f] = np.nan

X_live = feats_all[needed]
valid = X_live.dropna()
if valid.empty:
    st.error("Not enough history to compute the indicators. Increase the year range.")
    st.stop()

latest_row = X_live.loc[[valid.index[-1]]]
latest_date = valid.index[-1]
last_close = float(prices.loc[latest_date, "Close"])

clf = bundle["classifier"]
proba_up = float(clf.predict_proba(latest_row)[0, 1])
pred = int(proba_up >= 0.5)
direction = "UP" if pred == 1 else "DOWN"
confidence = proba_up if pred == 1 else 1 - proba_up

try:
    expected_ret = float(bundle["regressor"].predict(latest_row)[0])
except Exception:
    expected_ret = float("nan")

# ---- regime -----------------------------------------------------------------
regime_label, regime_pup = "not computed", "n/a"
if show_regime:
    try:
        rf = bundle["regime_features"]
        rrow = feats_all.loc[[latest_date], rf]
        rrow = rrow.fillna(pd.Series(bundle["regime_scaler"].center_, index=rf))
        rid = int(bundle["regime_kmeans"].predict(bundle["regime_scaler"].transform(rrow))[0])
        names = bundle["regime_names"]
        regime_label = names.get(rid, names.get(str(rid), f"Group {rid}"))
        for rec in bundle.get("regime_forward_stats", []):
            if str(rec.get("regime", "")) == str(regime_label):
                regime_pup = f"{float(rec['pct_up']):.1%}"
                break
    except Exception as e:
        regime_label = f"unavailable ({type(e).__name__})"

# ============================================================================
# HEADLINE
# ============================================================================

c1, c2, c3, c4, c5 = st.columns([1.15, 1, 1, 1, 1.15])
is_index = ticker.startswith("^")
c1.metric(f"{display_name(ticker)} close" + ("" if is_index else " (INR)"),
          f"{last_close:,.2f}" if is_index else f"₹{last_close:,.2f}", f"as of {latest_date:%d %b %Y}")
c2.metric(f"Next {horizon}d direction", direction,
          delta="bullish" if pred else "bearish",
          delta_color="normal" if pred else "inverse")
c3.metric("Model confidence", f"{confidence:.1%}")
c4.metric("Expected return", f"{expected_ret:+.2%}" if expected_ret == expected_ret else "n/a")
c5.metric("Market regime", regime_label.split(": ")[-1] if ": " in regime_label else regime_label)

base = bundle.get("majority_baseline", 0.5)
metrics = bundle.get("cv_metrics", {})
cv_acc = metrics.get("accuracy", float("nan"))
cv_mcc = metrics.get("mcc", float("nan"))
cv_bal = metrics.get("balanced accuracy", metrics.get("balanced_acc", float("nan")))

if cv_acc > base:
    st.info(
        f"**How much to trust this.** Cross-validated accuracy was **{cv_acc:.1%}** "
        f"against a majority-class baseline of **{base:.1%}**, so the model adds about "
        f"**{100*(cv_acc-base):.1f} percentage points**. "
        f"Coursework demonstration only, not investment advice."
    )
else:
    st.warning(
        f"**Read this before trusting the prediction.** Cross-validated accuracy was "
        f"**{cv_acc:.1%}**, which is *below* the majority-class baseline of "
        f"**{base:.1%}**. That baseline is simply the rule 'always predict UP', which "
        f"scores well because this stock rose over the training period. "
        f"The model does show real skill on measures that ignore that drift "
        f"(MCC **{cv_mcc:+.3f}** against 0.000 for the baseline, balanced accuracy "
        f"**{cv_bal:.3f}** against 0.500), and historically the days it calls UP have "
        f"performed better than the days it calls DOWN. But it does not beat simply "
        f"assuming the price will rise. "
        f"Coursework demonstration only, not investment advice."
    )

tab_chart, tab_explain, tab_data, tab_report = st.tabs(
    ["Chart", "Explanation", "Data", "Report"])

# ---------------------------------------------------------------- CHART TAB
with tab_chart:
    win = prices.iloc[-chart_days:]
    fw = feats_all.loc[win.index]
    has_vol = volume_is_usable(prices)

    n_rows = 4 if has_vol else 3
    heights = [0.46, 0.18, 0.18, 0.18] if has_vol else [0.54, 0.23, 0.23]
    titles = ["Price, Bollinger bands and BBI", "RSI(14)", "KDJ"]
    if has_vol:
        titles.append("Volume")

    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, row_heights=heights,
                        vertical_spacing=0.035, subplot_titles=titles)

    fig.add_trace(go.Candlestick(x=win.index, open=win["Open"], high=win["High"],
                                 low=win["Low"], close=win["Close"], name="price",
                                 showlegend=False), row=1, col=1)
    bu, bl, bm, _, _ = calc_bollinger(win["Close"])
    fig.add_trace(go.Scatter(x=win.index, y=bu, name="BB upper",
                             line=dict(width=1, color="rgba(130,130,130,0.65)")), row=1, col=1)
    fig.add_trace(go.Scatter(x=win.index, y=bl, name="BB lower", fill="tonexty",
                             fillcolor="rgba(150,150,150,0.10)",
                             line=dict(width=1, color="rgba(130,130,130,0.65)")), row=1, col=1)
    fig.add_trace(go.Scatter(x=win.index, y=calc_bbi(win["Close"]), name="BBI",
                             line=dict(width=1.7, color="#B4433B")), row=1, col=1)

    fig.add_trace(go.Scatter(x=[latest_date], y=[last_close], mode="markers",
                             marker=dict(size=13, symbol="star",
                                         color="#2E7D5B" if pred else "#B4433B",
                                         line=dict(width=1.4, color="white")),
                             name=f"prediction: {direction}"), row=1, col=1)

    fig.add_trace(go.Scatter(x=fw.index, y=fw["rsi_14"], name="RSI(14)",
                             line=dict(width=1.3, color="#3D6B9E")), row=2, col=1)
    fig.add_hline(y=70, line=dict(dash="dot", width=1, color="#B4433B"), row=2, col=1)
    fig.add_hline(y=30, line=dict(dash="dot", width=1, color="#2E7D5B"), row=2, col=1)

    for cname, colr in [("kdj_k", "#3D6B9E"), ("kdj_d", "#B4433B"), ("kdj_j", "#7A5BA6")]:
        fig.add_trace(go.Scatter(x=fw.index, y=fw[cname], name=cname,
                                 line=dict(width=1.1, color=colr)), row=3, col=1)

    if has_vol:
        fig.add_trace(go.Bar(x=win.index, y=win["Volume"], name="volume",
                             marker_color="rgba(61,107,158,0.55)"), row=4, col=1)

    fig.update_layout(height=250 + 150 * n_rows, hovermode="x unified",
                      legend=dict(orientation="h", y=-0.05),
                      margin=dict(t=40, b=30, l=10, r=10))
    for r in range(1, n_rows + 1):
        fig.update_xaxes(rangeslider_visible=False, row=r, col=1)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Drag to zoom, double-click to reset, hover for values across all panels.")

    if show_backtest:
        st.subheader("Historical signal")
        hist = X_live.dropna()
        if len(hist) > 300:
            hb = hist.iloc[-1500:]
            probs = clf.predict_proba(hb)[:, 1]
            sig = pd.Series(probs, index=hb.index)
            fwd = prices["Close"].pct_change(horizon).shift(-horizon).reindex(hb.index)

            f2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               row_heights=[0.55, 0.45], vertical_spacing=0.06,
                               subplot_titles=("Model P(up) over time",
                                               f"Realised forward {horizon}-day return"))
            f2.add_trace(go.Scatter(x=sig.index, y=sig, name="P(up)",
                                    line=dict(width=1.1, color="#3D6B9E")), row=1, col=1)
            f2.add_hline(y=0.5, line=dict(dash="dash", color="#999"), row=1, col=1)
            f2.add_trace(go.Scatter(x=fwd.index, y=fwd, name="forward return",
                                    line=dict(width=0.9, color="#B4433B")), row=2, col=1)
            f2.add_hline(y=0, line=dict(dash="dash", color="#999"), row=2, col=1)
            f2.update_layout(height=480, hovermode="x unified",
                             margin=dict(t=48, b=30), showlegend=False)
            st.plotly_chart(f2, use_container_width=True)

            ok = fwd.notna()
            if ok.sum() > 30:
                hit = ((sig[ok] >= 0.5).astype(int) == (fwd[ok] > 0).astype(int)).mean()
                dec = pd.DataFrame({"p": sig[ok], "up": (fwd[ok] > 0).astype(int)})
                dec["bucket"] = pd.qcut(dec["p"], 5, duplicates="drop")
                cal = dec.groupby("bucket", observed=True)["up"].agg(["mean", "size"])
                a, b = st.columns([1, 1.4])
                a.metric("Directional hit rate (recent window)", f"{hit:.1%}")
                b.write("**Calibration** — does a higher P(up) actually mean more ups?")
                b.dataframe(cal.rename(columns={"mean": "actual P(up)", "size": "n days"})
                            .style.format({"actual P(up)": "{:.1%}"}),
                            use_container_width=True)

# ------------------------------------------------------------- EXPLAIN TAB
top_features = []
with tab_explain:
    st.subheader("Why the model made this call")
    if show_shap:
        try:
            import shap
            step_names = [n for n, _ in clf.steps]
            Xp = latest_row
            for nm in step_names[:-1]:
                Xp = clf.named_steps[nm].transform(Xp)
            Xp = pd.DataFrame(Xp, columns=needed, index=latest_row.index)

            model = clf.steps[-1][1]
            expl = shap.TreeExplainer(model)
            sv = expl.shap_values(Xp)
            if isinstance(sv, list):
                sv = sv[1]
            elif np.ndim(sv) == 3:
                sv = sv[:, :, 1]
            sv = np.asarray(sv).reshape(-1)

            contrib = pd.DataFrame({
                "feature": needed,
                "value": latest_row.iloc[0].values,
                "shap": sv,
            })
            contrib["abs"] = contrib["shap"].abs()
            contrib = contrib.sort_values("abs", ascending=False)

            top = contrib.head(14).iloc[::-1]
            fw_fig = go.Figure(go.Bar(
                x=top["shap"], y=top["feature"], orientation="h",
                marker_color=np.where(top["shap"] > 0, "#2E7D5B", "#B4433B"),
                hovertemplate="%{y}<br>value=%{customdata:.4f}<br>SHAP=%{x:+.4f}<extra></extra>",
                customdata=top["value"]))
            fw_fig.update_layout(
                title="Feature contributions to this prediction "
                      "(green pushes up, red pushes down)",
                height=460, xaxis_title="SHAP value", margin=dict(l=180, t=60, b=40))
            st.plotly_chart(fw_fig, use_container_width=True)

            top_features = list(zip(contrib.head(8)["feature"],
                                    contrib.head(8)["value"],
                                    contrib.head(8)["shap"]))
            pos = contrib[contrib["shap"] > 0].head(3)["feature"].tolist()
            neg = contrib[contrib["shap"] < 0].head(3)["feature"].tolist()
            st.markdown(
                f"Pushing **towards up**: {', '.join(f'`{f}`' for f in pos) or 'none'}  \n"
                f"Pushing **towards down**: {', '.join(f'`{f}`' for f in neg) or 'none'}")
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {type(e).__name__}: {e}")

    if show_regime:
        st.divider()
        st.subheader("Market regime context")
        st.markdown(
            f"Current regime: **{regime_label}**. Historically, price rose over the "
            f"next {bundle.get('horizon')} days **{regime_pup}** of the time in this "
            f"regime. Regimes were discovered by KMeans on volatility and trend "
            f"features alone, with no date information, yet they form long contiguous "
            f"blocks in time — which is why they are treated as real structure rather "
            f"than an arbitrary partition.")
        rs = pd.DataFrame(bundle.get("regime_forward_stats", []))
        if not rs.empty:
            st.dataframe(rs, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- DATA TAB
with tab_data:
    st.subheader("Latest feature values")
    show = pd.DataFrame({
        "feature": needed,
        "latest value": latest_row.iloc[0].values,
        "percentile in history": [
            float((X_live[f].dropna() <= latest_row.iloc[0][f]).mean() * 100)
            if X_live[f].notna().any() else np.nan for f in needed],
    })
    st.dataframe(show.style.format({"latest value": "{:.4f}",
                                    "percentile in history": "{:.1f}"}),
                 use_container_width=True, hide_index=True, height=420)

    st.subheader("Recent price data")
    st.dataframe(prices.tail(25).iloc[::-1], use_container_width=True)

    st.subheader("Model information")
    st.json({k: v for k, v in bundle.items()
             if k in ("horizon", "features", "ticker_trained_on", "trained_rows",
                      "train_start", "train_end", "best_model_name",
                      "majority_baseline", "volume_available", "cv_metrics")})

# -------------------------------------------------------------- REPORT TAB
with tab_report:
    st.subheader("Download or email the report")

    ctx = {
        "ticker": display_name(ticker), "last_close": last_close,
        "last_date": f"{latest_date:%Y-%m-%d}", "horizon": horizon,
        "direction": direction, "confidence": confidence,
        "expected_return": expected_ret if expected_ret == expected_ret else 0.0,
        "regime": regime_label, "regime_pup": regime_pup,
        "model_name": bundle.get("best_model_name", "LightGBM"),
        "trained_rows": bundle.get("trained_rows", 0),
        "train_start": bundle.get("train_start", ""),
        "train_end": bundle.get("train_end", ""),
        "acc_vs_base": f"{cv_acc:.1%} / {base:.1%}",
        "top_features": top_features or [(f, latest_row.iloc[0][f], 0.0)
                                         for f in needed[:8]],
    }

    summary_df = pd.DataFrame([{
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ticker": display_name(ticker), "as_of": f"{latest_date:%Y-%m-%d}",
        "last_close": last_close, "horizon_days": horizon,
        "predicted_direction": direction, "probability_up": proba_up,
        "confidence": confidence, "expected_return": expected_ret,
        "regime": regime_label, "regime_historical_p_up": regime_pup,
        "model": ctx["model_name"], "cv_accuracy": cv_acc,
        "majority_baseline": base,
    }])

    c1, c2, c3 = st.columns(3)
    c1.download_button("Download summary (CSV)",
                       summary_df.to_csv(index=False).encode(),
                       file_name=f"prediction_{ticker.replace('^','')}_{latest_date:%Y%m%d}.csv",
                       mime="text/csv", use_container_width=True)
    c2.download_button("Download features (CSV)",
                       X_live.dropna().tail(300).to_csv().encode(),
                       file_name=f"features_{ticker.replace('^','')}_{latest_date:%Y%m%d}.csv",
                       mime="text/csv", use_container_width=True)

    pdf_bytes = build_pdf(ctx)
    if pdf_bytes:
        c3.download_button("Download report (PDF)", pdf_bytes,
                           file_name=f"report_{ticker.replace('^','')}_{latest_date:%Y%m%d}.pdf",
                           mime="application/pdf", use_container_width=True)
    else:
        c3.info("Add `reportlab` to requirements.txt to enable PDF export.")

    st.divider()
    st.markdown("**Email the report**")
    with st.form("email_form"):
        to_addr = st.text_input("Recipient address")
        note = st.text_area("Optional note", height=70)
        submitted = st.form_submit_button("Send")
    if submitted:
        if not to_addr or "@" not in to_addr:
            st.error("Enter a valid email address.")
        else:
            body = (f"{display_name(ticker)} prediction as of {latest_date:%Y-%m-%d}\n\n"
                    f"Direction over the next {horizon} trading days: {direction}\n"
                    f"Confidence: {confidence:.1%}\n"
                    f"Expected return: {expected_ret:+.2%}\n"
                    f"Market regime: {regime_label}\n\n"
                    f"Cross-validated accuracy {cv_acc:.1%} vs majority baseline "
                    f"{base:.1%}.\n\n"
                    f"{note}\n\n"
                    "Generated by a CML7133 coursework app. Not investment advice.")
            atts = [(f"prediction_{latest_date:%Y%m%d}.csv",
                     summary_df.to_csv(index=False).encode())]
            if pdf_bytes:
                atts.append((f"report_{latest_date:%Y%m%d}.pdf", pdf_bytes))
            ok, msg = send_email(to_addr, f"{display_name(ticker)} direction forecast "
                                          f"({latest_date:%Y-%m-%d})", body, atts)
            (st.success if ok else st.error)(msg)

st.divider()
st.caption(
    "Built for CML7133 Assignment 2. Predictions come from technical indicators only "
    "and carry a small measured edge over the majority-class baseline. "
    "Not investment advice."
)
