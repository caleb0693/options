
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, time
from zoneinfo import ZoneInfo

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None


# ============================================================
# AUTOMATIC EVENT LOGGER UPGRADE
# Add this to the app
# ============================================================

import os

LOG_FILE = "spx_event_log.csv"


def log_event(
    timestamp,
    price,
    regime,
    score,
    day_type,
    breadth,
    breadth_count,
    rvol,
    signal,
    decision,
    decision_reason,
    gate_df
):
    gate_summary = " | ".join([
        f"{row['Gate']}: {row['Status']}"
        for _, row in gate_df.iterrows()
    ])

    event_row = pd.DataFrame([{
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "price": round(price, 2),
        "regime": regime,
        "regime_score": score,
        "day_type": day_type,
        "breadth": round(breadth, 2),
        "breadth_persistence": breadth_count,
        "rvol": round(rvol, 2) if pd.notna(rvol) else np.nan,
        "signal": signal,
        "decision": decision,
        "decision_reason": decision_reason,
        "gate_status": gate_summary
    }])

    file_exists = os.path.exists(LOG_FILE)

    event_row.to_csv(
        LOG_FILE,
        mode="a",
        header=not file_exists,
        index=False
    )



# ============================================================
# App Configuration
# ============================================================

st.set_page_config(
    page_title="SPX 0DTE Gated Decision Dashboard",
    page_icon="📈",
    layout="wide"
)

EASTERN = ZoneInfo("America/New_York")

ETF_SYMBOLS = [
    "SPY", "QQQ", "IWM",
    "XLK", "XLF", "XLV", "XLY", "XLI",
    "XLE", "XLP", "XLU", "XLB", "XLRE", "XLC"
]


# ============================================================
# Market Data Functions
# ============================================================

@st.cache_data(ttl=30)
def get_intraday_data(symbol: str, period: str = "1d", interval: str = "1m") -> pd.DataFrame:
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            prepost=False
        )
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()

    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "datetime"})
    elif "Date" in df.columns:
        df = df.rename(columns={"Date": "datetime"})

    df["datetime"] = pd.to_datetime(df["datetime"])

    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC").dt.tz_convert(EASTERN)
    else:
        df["datetime"] = df["datetime"].dt.tz_convert(EASTERN)

    return df


@st.cache_data(ttl=300)
def get_daily_data(symbol: str, period: str = "30d") -> pd.DataFrame:
    try:
        df = yf.download(
            symbol,
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=False
        )
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    return df.reset_index()


@st.cache_data(ttl=300)
def get_multiday_intraday_data(symbol: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            prepost=False
        )
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()

    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "datetime"})
    elif "Date" in df.columns:
        df = df.rename(columns={"Date": "datetime"})

    df["datetime"] = pd.to_datetime(df["datetime"])

    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC").dt.tz_convert(EASTERN)
    else:
        df["datetime"] = df["datetime"].dt.tz_convert(EASTERN)

    return df


def calculate_vwap(df: pd.DataFrame) -> float:
    if df.empty or "Volume" not in df.columns or df["Volume"].sum() == 0:
        return np.nan

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    return float((typical_price * df["Volume"]).sum() / df["Volume"].sum())


def calculate_ema(df: pd.DataFrame, span: int) -> float:
    if df.empty or "Close" not in df.columns:
        return np.nan

    return float(df["Close"].ewm(span=span, adjust=False).mean().iloc[-1])


def calculate_ema_slope(df: pd.DataFrame, span: int = 20, lookback: int = 5) -> float:
    if df.empty or len(df) < span + lookback:
        return np.nan

    ema = df["Close"].ewm(span=span, adjust=False).mean()
    return float(ema.iloc[-1] - ema.iloc[-lookback])


def calculate_atr(df: pd.DataFrame, window: int = 14) -> float:
    if df.empty or len(df) < window + 1:
        return np.nan

    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    return float(tr.rolling(window).mean().iloc[-1])


def get_previous_day_levels(symbol: str):
    daily = get_daily_data(symbol)

    if daily.empty or len(daily) < 2:
        return np.nan, np.nan, np.nan

    prev = daily.iloc[-2]
    return float(prev["High"]), float(prev["Low"]), float(prev["Close"])


def get_opening_range(df: pd.DataFrame):
    if df.empty:
        return np.nan, np.nan

    temp = df.copy()
    temp["t"] = temp["datetime"].dt.time

    opening = temp[
        (temp["t"] >= time(9, 30)) &
        (temp["t"] < time(9, 45))
    ]

    if opening.empty:
        return np.nan, np.nan

    return float(opening["High"].max()), float(opening["Low"].min())


def market_time_now():
    return datetime.now(EASTERN)


# ============================================================
# Filters and Classifiers
# ============================================================

def calculate_rvol(today_5m_df: pd.DataFrame, history_5m_df: pd.DataFrame):
    if today_5m_df.empty or history_5m_df.empty or "Volume" not in today_5m_df.columns:
        return np.nan, "RVOL unavailable"

    now_t = today_5m_df["datetime"].iloc[-1].time()
    today_date = today_5m_df["datetime"].dt.date.iloc[-1]
    current_cum_vol = today_5m_df["Volume"].sum()

    hist = history_5m_df.copy()
    hist["date"] = hist["datetime"].dt.date
    hist["t"] = hist["datetime"].dt.time

    prior = hist[hist["date"] != today_date]
    prior_same_time = prior[prior["t"] <= now_t]

    if prior_same_time.empty:
        return np.nan, "RVOL unavailable"

    daily_cum = prior_same_time.groupby("date")["Volume"].sum()

    if daily_cum.empty or daily_cum.mean() == 0:
        return np.nan, "RVOL unavailable"

    rvol = float(current_cum_vol / daily_cum.mean())

    if rvol >= 1.5:
        label = "High relative volume"
    elif rvol >= 1.0:
        label = "Normal/elevated volume"
    elif rvol >= 0.8:
        label = "Slightly weak volume"
    else:
        label = "Weak relative volume"

    return rvol, label


def classify_gap(open_price, prev_close, prev_high, prev_low):
    if any(pd.isna(x) for x in [open_price, prev_close, prev_high, prev_low]):
        return "Gap unavailable", np.nan

    gap_pct = ((open_price - prev_close) / prev_close) * 100

    if open_price > prev_high:
        gap_type = "Gap Up Above Prior High"
    elif open_price < prev_low:
        gap_type = "Gap Down Below Prior Low"
    elif abs(gap_pct) < 0.15:
        gap_type = "Flat / Small Gap"
    elif open_price > prev_close:
        gap_type = "Inside-Range Gap Up"
    else:
        gap_type = "Inside-Range Gap Down"

    return gap_type, float(gap_pct)


def classify_day_type(price, vwap, opening_high, opening_low, atr_5m, breadth, rvol, ema_slope):
    if any(pd.isna(x) for x in [price, vwap, opening_high, opening_low, atr_5m, breadth]):
        return "Unknown", ["Missing day-type inputs"]

    reasons = []

    vwap_distance = abs(price - vwap)
    strong_breadth = abs(breadth) >= 0.60
    high_rvol = pd.notna(rvol) and rvol >= 1.2
    strong_ema_slope = pd.notna(ema_slope) and abs(ema_slope) >= atr_5m * 0.25
    outside_opening_range = price > opening_high or price < opening_low
    extended_from_vwap = vwap_distance >= atr_5m

    reasons.append("Price outside opening range" if outside_opening_range else "Price inside opening range")
    reasons.append("Strong breadth" if strong_breadth else "Breadth not strong")
    reasons.append("RVOL supportive" if high_rvol else "RVOL not elevated")
    reasons.append("EMA slope supportive" if strong_ema_slope else "EMA slope not strong")

    if outside_opening_range and strong_breadth and (high_rvol or strong_ema_slope):
        return "Trend / Momentum Day", reasons

    if outside_opening_range and extended_from_vwap and not strong_breadth:
        return "Breakout Risk / Possible Fade", reasons

    if not outside_opening_range and not strong_breadth and not high_rvol:
        return "Chop / Range Day", reasons

    if outside_opening_range and high_rvol:
        return "Expansion Day", reasons

    return "Neutral / Developing", reasons


def update_breadth_persistence(current_breadth, threshold=0.60):
    if "breadth_persistence" not in st.session_state:
        st.session_state["breadth_persistence"] = {
            "direction": "Neutral",
            "count": 0,
            "history": []
        }

    if current_breadth >= threshold:
        direction = "Bullish"
    elif current_breadth <= -threshold:
        direction = "Bearish"
    else:
        direction = "Neutral"

    state = st.session_state["breadth_persistence"]

    if direction == state["direction"] and direction != "Neutral":
        state["count"] += 1
    elif direction != "Neutral":
        state["direction"] = direction
        state["count"] = 1
    else:
        state["direction"] = "Neutral"
        state["count"] = 0

    state["history"].append({
        "timestamp": datetime.now(EASTERN).strftime("%H:%M:%S"),
        "breadth": current_breadth,
        "direction": direction,
        "count": state["count"]
    })

    state["history"] = state["history"][-50:]
    st.session_state["breadth_persistence"] = state

    return state["direction"], state["count"], pd.DataFrame(state["history"])


@st.cache_data(ttl=30)
def calculate_auto_etf_breadth():
    rows = []
    positive = 0
    negative = 0

    for symbol in ETF_SYMBOLS:
        df = get_intraday_data(symbol)

        if df.empty:
            continue

        last_price = float(df["Close"].iloc[-1])
        vwap = calculate_vwap(df)

        if pd.isna(vwap):
            continue

        above_vwap = last_price > vwap

        if above_vwap:
            positive += 1
        else:
            negative += 1

        pct_from_vwap = ((last_price - vwap) / vwap) * 100

        rows.append({
            "Symbol": symbol,
            "Last": round(last_price, 2),
            "VWAP": round(vwap, 2),
            "Above VWAP": above_vwap,
            "% From VWAP": round(pct_from_vwap, 2)
        })

    total = positive + negative
    breadth = 0.0 if total == 0 else (positive - negative) / total

    return breadth, positive, negative, pd.DataFrame(rows)


def calculate_manual_breadth_score(advancers, decliners):
    total = advancers + decliners

    if total == 0:
        return 0.0

    return (advancers - decliners) / total


def normalize_tick(tick_value):
    return max(min(tick_value / 1000, 1), -1)


def normalize_vold(vold_value):
    return max(min(vold_value / 1_000_000_000, 1), -1)


def composite_manual_breadth(advancers, decliners, tick_value, vold_value):
    adv_decl_score = calculate_manual_breadth_score(advancers, decliners)
    tick_score = normalize_tick(tick_value)
    vold_score = normalize_vold(vold_value)

    composite = (
        0.50 * adv_decl_score +
        0.30 * tick_score +
        0.20 * vold_score
    )

    return composite, adv_decl_score, tick_score, vold_score


# ============================================================
# Regime, Signal, Risk, and Gated Decision Flow
# ============================================================

def classify_regime(price, vwap, ema9, ema20, prev_high, prev_low, vix, breadth):
    score = 0
    reasons = []

    if pd.notna(vwap):
        if price > vwap:
            score += 1
            reasons.append("Price above VWAP")
        else:
            score -= 1
            reasons.append("Price below VWAP")
    else:
        reasons.append("VWAP unavailable")

    if pd.notna(ema9) and pd.notna(ema20):
        if ema9 > ema20:
            score += 1
            reasons.append("9 EMA above 20 EMA")
        else:
            score -= 1
            reasons.append("9 EMA below 20 EMA")
    else:
        reasons.append("EMA data unavailable")

    if pd.notna(prev_high) and pd.notna(prev_low):
        if price > prev_high:
            score += 1
            reasons.append("Price above previous day high")
        elif price < prev_low:
            score -= 1
            reasons.append("Price below previous day low")
        else:
            reasons.append("Price inside prior day range")
    else:
        reasons.append("Previous day levels unavailable")

    if breadth > 0.60:
        score += 1
        reasons.append("Strong bullish breadth")
    elif breadth < -0.60:
        score -= 1
        reasons.append("Strong bearish breadth")
    else:
        reasons.append("Neutral / mixed breadth")

    if pd.notna(vix):
        if vix > 25:
            reasons.append("High VIX: larger moves and higher option premiums")
        elif vix < 13:
            reasons.append("Low VIX: option premium may decay quickly")
        else:
            reasons.append("Moderate VIX")
    else:
        reasons.append("VIX unavailable")

    if score >= 3:
        regime = "Bullish trend / call-favored"
    elif score <= -3:
        regime = "Bearish trend / put-favored"
    else:
        regime = "Mixed / no-trade or wait"

    return regime, score, reasons


def trade_signal(regime, price, opening_high, opening_low, vwap, atr_5m, current_time):
    signal = "No Trade"
    setup = "Wait for confirmation"
    invalidation = "No valid setup"

    in_morning_window = time(9, 35) <= current_time <= time(10, 30)
    in_afternoon_window = time(13, 30) <= current_time <= time(15, 0)

    if not (in_morning_window or in_afternoon_window):
        return signal, setup, "Outside preferred trading window"

    if any(pd.isna(x) for x in [opening_high, opening_low, vwap, atr_5m]):
        return signal, setup, "Missing required market inputs"

    if "call-favored" in regime and price > opening_high and price > vwap:
        signal = "CALL Watch"
        setup = "Opening range / trend continuation breakout"
        invalidation = max(vwap, opening_high - atr_5m)

    elif "put-favored" in regime and price < opening_low and price < vwap:
        signal = "PUT Watch"
        setup = "Opening range / trend continuation breakdown"
        invalidation = min(vwap, opening_low + atr_5m)

    return signal, setup, invalidation


def risk_box(account_size, risk_pct, option_price, stop_loss_pct):
    risk_dollars = account_size * risk_pct / 100
    risk_per_contract = option_price * 100 * stop_loss_pct / 100

    if risk_per_contract <= 0:
        contracts = 0
    else:
        contracts = int(risk_dollars // risk_per_contract)

    max_premium = contracts * option_price * 100

    return risk_dollars, risk_per_contract, contracts, max_premium


def gated_trade_decision(
    signal,
    score,
    price,
    vwap,
    opening_high,
    opening_low,
    breadth,
    breadth_count,
    rvol,
    day_type,
    contracts,
    current_time,
    invalidation
):
    flow_rows = []

    def add_gate(gate, status, detail):
        flow_rows.append({
            "Gate": gate,
            "Status": status,
            "Detail": detail
        })

    # 1. Environment Gate
    tradable_window = (
        time(9, 35) <= current_time <= time(10, 30)
        or time(13, 30) <= current_time <= time(15, 0)
    )

    if not tradable_window:
        add_gate("1. Environment", "FAIL", "Outside preferred trading window.")
        return "STAY OUT", "Environment gate failed: outside preferred trading window.", pd.DataFrame(flow_rows)

    if day_type in ["Chop / Range Day", "Breakout Risk / Possible Fade"]:
        add_gate("1. Environment", "FAIL", f"Day type is {day_type}.")
        return "STAY OUT", f"Environment gate failed: {day_type}.", pd.DataFrame(flow_rows)

    if pd.notna(rvol) and rvol < 0.80:
        add_gate("1. Environment", "FAIL", f"RVOL is weak at {rvol:.2f}.")
        return "STAY OUT", "Environment gate failed: weak relative volume.", pd.DataFrame(flow_rows)

    add_gate("1. Environment", "PASS", f"Day type is {day_type}; RVOL acceptable.")

    # 2. Direction Gate
    if signal == "CALL Watch":
        if score < 3:
            add_gate("2. Direction", "FAIL", f"CALL requires regime score >= +3. Current score: {score}.")
            return "STAY OUT", "Direction gate failed: bullish regime not strong enough.", pd.DataFrame(flow_rows)

        if breadth <= 0:
            add_gate("2. Direction", "FAIL", f"CALL requires positive breadth. Current breadth: {breadth:.2f}.")
            return "STAY OUT", "Direction gate failed: breadth does not support CALL.", pd.DataFrame(flow_rows)

        if price <= vwap:
            add_gate("2. Direction", "FAIL", "CALL requires price above VWAP.")
            return "STAY OUT", "Direction gate failed: price not above VWAP.", pd.DataFrame(flow_rows)

        add_gate("2. Direction", "PASS", "Bullish regime, positive breadth, price above VWAP.")

    elif signal == "PUT Watch":
        if score > -3:
            add_gate("2. Direction", "FAIL", f"PUT requires regime score <= -3. Current score: {score}.")
            return "STAY OUT", "Direction gate failed: bearish regime not strong enough.", pd.DataFrame(flow_rows)

        if breadth >= 0:
            add_gate("2. Direction", "FAIL", f"PUT requires negative breadth. Current breadth: {breadth:.2f}.")
            return "STAY OUT", "Direction gate failed: breadth does not support PUT.", pd.DataFrame(flow_rows)

        if price >= vwap:
            add_gate("2. Direction", "FAIL", "PUT requires price below VWAP.")
            return "STAY OUT", "Direction gate failed: price not below VWAP.", pd.DataFrame(flow_rows)

        add_gate("2. Direction", "PASS", "Bearish regime, negative breadth, price below VWAP.")

    else:
        add_gate("2. Direction", "FAIL", "No CALL Watch or PUT Watch signal.")
        return "STAY OUT", "Direction gate failed: no actionable directional signal.", pd.DataFrame(flow_rows)

    # 3. Structure Gate
    if signal == "CALL Watch":
        if price <= opening_high:
            add_gate("3. Structure", "FAIL", "CALL requires price above opening range high.")
            return "STAY OUT", "Structure gate failed: no confirmed opening range breakout.", pd.DataFrame(flow_rows)

        add_gate("3. Structure", "PASS", "Price is above opening range high.")

    elif signal == "PUT Watch":
        if price >= opening_low:
            add_gate("3. Structure", "FAIL", "PUT requires price below opening range low.")
            return "STAY OUT", "Structure gate failed: no confirmed opening range breakdown.", pd.DataFrame(flow_rows)

        add_gate("3. Structure", "PASS", "Price is below opening range low.")

    # 4. Persistence Gate
    if signal == "CALL Watch":
        if 0 < breadth < 0.60:
            add_gate("4. Persistence", "CAUTION", f"Breadth positive but not strong: {breadth:.2f}.")
            return "WATCH ONLY", "Watch only: CALL setup exists, but breadth is not strong.", pd.DataFrame(flow_rows)

        if breadth >= 0.60 and breadth_count < 1:
            add_gate("4. Persistence", "WAIT", "Strong bullish breadth has not persisted yet.")
            return "WATCH ONLY", "Wait for bullish breadth persistence.", pd.DataFrame(flow_rows)

        add_gate("4. Persistence", "PASS", "Bullish breadth is strong and persistent enough.")

    elif signal == "PUT Watch":
        if -0.60 < breadth < 0:
            add_gate("4. Persistence", "CAUTION", f"Breadth negative but not strong: {breadth:.2f}.")
            return "WATCH ONLY", "Watch only: PUT setup exists, but breadth is not strong.", pd.DataFrame(flow_rows)

        if breadth <= -0.60 and breadth_count < 1:
            add_gate("4. Persistence", "WAIT", "Strong bearish breadth has not persisted yet.")
            return "WATCH ONLY", "Wait for bearish breadth persistence.", pd.DataFrame(flow_rows)

        add_gate("4. Persistence", "PASS", "Bearish breadth is strong and persistent enough.")

    # 5. Risk Gate
    if contracts <= 0:
        add_gate("5. Risk", "FAIL", "Position size returned zero contracts.")
        return "STAY OUT", "Risk gate failed: position size is not valid.", pd.DataFrame(flow_rows)

    if not isinstance(invalidation, (int, float, np.floating)):
        add_gate("5. Risk", "FAIL", "No defined invalidation level.")
        return "STAY OUT", "Risk gate failed: no defined invalidation level.", pd.DataFrame(flow_rows)

    add_gate("5. Risk", "PASS", f"Contracts: {contracts}; invalidation defined.")

    # 6. Final Decision
    if signal == "CALL Watch":
        add_gate("6. Final Decision", "PASS", "All gates passed for BUY CALL.")
        return "BUY CALL", "All gates passed: bullish environment, direction, structure, persistence, and risk.", pd.DataFrame(flow_rows)

    if signal == "PUT Watch":
        add_gate("6. Final Decision", "PASS", "All gates passed for BUY PUT.")
        return "BUY PUT", "All gates passed: bearish environment, direction, structure, persistence, and risk.", pd.DataFrame(flow_rows)

    add_gate("6. Final Decision", "FAIL", "No actionable setup.")
    return "STAY OUT", "No actionable setup after gated evaluation.", pd.DataFrame(flow_rows)



# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("SPX 0DTE Controls")

st.sidebar.subheader("Refresh Controls")

auto_refresh_enabled = st.sidebar.toggle("Enable Auto Refresh", value=True)

refresh_seconds = st.sidebar.selectbox(
    "Refresh interval (seconds)",
    [15, 30, 60, 120, 300],
    index=1
)

manual_refresh = st.sidebar.button("Refresh Now")

if manual_refresh:
    st.cache_data.clear()
    st.rerun()

if auto_refresh_enabled and st_autorefresh is not None:
    st_autorefresh(interval=refresh_seconds * 1000, key="market_refresh")

st.sidebar.divider()

account_size = st.sidebar.number_input(
    "Account Size ($)",
    min_value=500.0,
    value=10000.0,
    step=500.0
)

risk_pct = st.sidebar.slider(
    "Risk per Trade (%)",
    0.25,
    5.0,
    1.0,
    0.25
)

max_daily_loss_pct = st.sidebar.slider(
    "Max Daily Loss (%)",
    1.0,
    10.0,
    3.0,
    0.5
)

st.sidebar.divider()

st.sidebar.subheader("Breadth Mode")

breadth_mode = st.sidebar.radio(
    "Select breadth source",
    [
        "Auto ETF Breadth Proxy",
        "Manual Composite Internals",
        "Simple Manual Score"
    ],
    index=0
)

if breadth_mode == "Auto ETF Breadth Proxy":
    breadth, positive, negative, breadth_df = calculate_auto_etf_breadth()
    breadth_source = "Auto ETF Breadth Proxy"

    st.sidebar.metric("Breadth Score", f"{breadth:.2f}")
    st.sidebar.metric("ETFs Above VWAP", positive)
    st.sidebar.metric("ETFs Below VWAP", negative)

elif breadth_mode == "Manual Composite Internals":
    st.sidebar.caption("Use values from TradingView, broker charts, or a market internals dashboard.")

    advancers = st.sidebar.number_input("Advancers", min_value=0, value=320, step=1)
    decliners = st.sidebar.number_input("Decliners", min_value=0, value=180, step=1)
    tick_value = st.sidebar.number_input("TICK", value=0, step=50)
    vold_value = st.sidebar.number_input("VOLD", value=0, step=50_000_000)

    breadth, adv_decl_score, tick_score, vold_score = composite_manual_breadth(
        advancers,
        decliners,
        tick_value,
        vold_value
    )

    breadth_source = "Manual Composite Internals"
    breadth_df = pd.DataFrame()

    st.sidebar.metric("Breadth Score", f"{breadth:.2f}")
    st.sidebar.metric("ADV/DECL Score", f"{adv_decl_score:.2f}")
    st.sidebar.metric("TICK Score", f"{tick_score:.2f}")
    st.sidebar.metric("VOLD Score", f"{vold_score:.2f}")

else:
    breadth = st.sidebar.slider("Manual Breadth Score", -1.0, 1.0, 0.0, 0.1)
    breadth_source = "Simple Manual Score"
    breadth_df = pd.DataFrame()
    positive = negative = 0

if breadth >= 0.60:
    st.sidebar.success("Strong bullish breadth")
elif breadth <= -0.60:
    st.sidebar.error("Strong bearish breadth")
else:
    st.sidebar.warning("Neutral / mixed breadth")

st.sidebar.divider()

st.sidebar.subheader("Option Contract")

option_price = st.sidebar.number_input(
    "Option Premium ($)",
    min_value=0.05,
    value=8.00,
    step=0.05
)

stop_loss_pct = st.sidebar.slider(
    "Stop Loss on Option (%)",
    10,
    60,
    25,
    5
)

target_pct = st.sidebar.slider(
    "Profit Target on Option (%)",
    10,
    100,
    30,
    5
)

st.sidebar.divider()

manual_market_override = st.sidebar.toggle("Manual market override", value=False)


# ============================================================
# Market Data Pull
# ============================================================

now_et = market_time_now()

if not manual_market_override:
    underlying_symbol = "SPY"
    vix_symbol = "^VIX"

    price_df = get_intraday_data(underlying_symbol)
    price_5m_df = get_intraday_data(underlying_symbol, interval="5m")
    history_5m_df = get_multiday_intraday_data(underlying_symbol, period="5d", interval="5m")
    vix_df = get_intraday_data(vix_symbol)

    if price_df.empty:
        st.error("Could not load SPY data from yfinance. Try manual market override.")
        st.stop()

    price = float(price_df["Close"].iloc[-1])
    open_price = float(price_df["Open"].iloc[0])
    last_timestamp = price_df["datetime"].iloc[-1]

    vix = float(vix_df["Close"].iloc[-1]) if not vix_df.empty else np.nan

    prev_high, prev_low, prev_close = get_previous_day_levels(underlying_symbol)
    opening_high, opening_low = get_opening_range(price_df)

    vwap = calculate_vwap(price_df)
    ema9 = calculate_ema(price_df, 9)
    ema20 = calculate_ema(price_df, 20)
    ema_slope = calculate_ema_slope(price_df, span=20, lookback=5)

    atr_5m = calculate_atr(price_5m_df, 14)

    rvol, rvol_label = calculate_rvol(price_5m_df, history_5m_df)
    gap_type, gap_pct = classify_gap(open_price, prev_close, prev_high, prev_low)

    current_time = now_et.time()

else:
    st.sidebar.subheader("Manual Market Inputs")

    price = st.sidebar.number_input("Current Price / SPY Proxy", value=530.0, step=0.1)
    open_price = st.sidebar.number_input("Open Price", value=530.0, step=0.1)
    prev_high = st.sidebar.number_input("Previous Day High", value=532.0, step=0.1)
    prev_low = st.sidebar.number_input("Previous Day Low", value=526.5, step=0.1)
    prev_close = st.sidebar.number_input("Previous Day Close", value=529.0, step=0.1)
    opening_high = st.sidebar.number_input("Opening Range High", value=531.0, step=0.1)
    opening_low = st.sidebar.number_input("Opening Range Low", value=528.5, step=0.1)
    vwap = st.sidebar.number_input("VWAP", value=529.8, step=0.1)
    ema9 = st.sidebar.number_input("9 EMA", value=530.5, step=0.1)
    ema20 = st.sidebar.number_input("20 EMA", value=529.5, step=0.1)
    ema_slope = st.sidebar.number_input("EMA Slope Proxy", value=0.2, step=0.1)
    atr_5m = st.sidebar.number_input("5-Min ATR", value=0.6, step=0.1)
    vix = st.sidebar.number_input("VIX", value=16.0, step=0.5)
    rvol = st.sidebar.number_input("RVOL", value=1.0, step=0.1)
    rvol_label = "Manual RVOL"
    current_time = st.sidebar.time_input("Market Time", value=time(9, 45))
    last_timestamp = now_et
    price_df = pd.DataFrame()
    gap_type, gap_pct = classify_gap(open_price, prev_close, prev_high, prev_low)


# ============================================================
# Derived Classifiers
# ============================================================

breadth_direction, breadth_count, breadth_history_df = update_breadth_persistence(
    current_breadth=breadth,
    threshold=0.60
)

day_type, day_type_reasons = classify_day_type(
    price=price,
    vwap=vwap,
    opening_high=opening_high,
    opening_low=opening_low,
    atr_5m=atr_5m,
    breadth=breadth,
    rvol=rvol,
    ema_slope=ema_slope
)

regime, score, reasons = classify_regime(
    price=price,
    vwap=vwap,
    ema9=ema9,
    ema20=ema20,
    prev_high=prev_high,
    prev_low=prev_low,
    vix=vix,
    breadth=breadth
)

signal, setup, invalidation = trade_signal(
    regime=regime,
    price=price,
    opening_high=opening_high,
    opening_low=opening_low,
    vwap=vwap,
    atr_5m=atr_5m,
    current_time=current_time
)

risk_dollars, risk_per_contract, contracts, max_premium = risk_box(
    account_size=account_size,
    risk_pct=risk_pct,
    option_price=option_price,
    stop_loss_pct=stop_loss_pct
)

daily_max_loss = account_size * max_daily_loss_pct / 100
profit_target_price = option_price * (1 + target_pct / 100)
stop_price = option_price * (1 - stop_loss_pct / 100)

decision, decision_reason, gate_df = gated_trade_decision(
    signal=signal,
    score=score,
    price=price,
    vwap=vwap,
    opening_high=opening_high,
    opening_low=opening_low,
    breadth=breadth,
    breadth_count=breadth_count,
    rvol=rvol,
    day_type=day_type,
    contracts=contracts,
    current_time=current_time,
    invalidation=invalidation
)


# ============================================================
# Main App
# ============================================================

st.title("SPX 0DTE Gated Decision Dashboard")
st.caption("Uses SPY as a liquid proxy for SPX directional logic. Decision support only. Does not execute trades.")

with st.expander("Decision Flow", expanded=False):
    st.markdown("""
    The app now uses a gated decision flow:

    1. **Environment Gate**: checks time window, day type, and RVOL.
    2. **Direction Gate**: checks regime score, breadth direction, and VWAP alignment.
    3. **Structure Gate**: checks opening range breakout/breakdown.
    4. **Persistence Gate**: checks whether breadth is strong enough or persistent enough.
    5. **Risk Gate**: checks position size and invalidation.
    6. **Final Decision**: outputs BUY CALL, BUY PUT, WATCH ONLY, or STAY OUT.

    This is designed to prevent aggressive trade signals when the broader context does not support the move.
    """)

top1, top2, top3, top4 = st.columns(4)

top1.metric("SPY Proxy", f"{price:,.2f}")
top2.metric("VIX", "N/A" if pd.isna(vix) else f"{vix:,.2f}")
top3.metric("Breadth", f"{breadth:.2f}")
top4.metric("Last Bar", str(last_timestamp).split(".")[0])

st.caption(f"Breadth Source: {breadth_source}")

if breadth_mode == "Auto ETF Breadth Proxy":
    st.caption(f"ETF Breadth: {positive} above VWAP / {negative} below VWAP")

st.divider()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Regime", regime)
c2.metric("Regime Score", score)
c3.metric("Signal", signal)
c4.metric("Suggested Contracts", contracts)

u1, u2, u3, u4 = st.columns(4)

u1.metric("Day Type", day_type)
u2.metric("RVOL", "N/A" if pd.isna(rvol) else f"{rvol:.2f}", rvol_label)
u3.metric("Gap Type", gap_type)
u4.metric("Breadth Persistence", f"{breadth_direction} x{breadth_count}")

st.divider()

st.subheader("Final Trade Decision")

if decision == "BUY CALL":
    st.success(f"### {decision}")
elif decision == "BUY PUT":
    st.error(f"### {decision}")
elif decision == "WATCH ONLY":
    st.info(f"### {decision}")
else:
    st.warning(f"### {decision}")

st.write(f"**Reason:** {decision_reason}")

d1, d2, d3, d4 = st.columns(4)

d1.metric("Contracts", contracts)
d2.metric("Entry Ref.", f"${option_price:,.2f}")
d3.metric("Stop", f"${stop_price:,.2f}")
d4.metric("Target", f"${profit_target_price:,.2f}")

st.caption("This command is a rules-based decision aid only. It does not execute trades.")

st.divider()

st.subheader("Gated Decision Flow")
st.dataframe(gate_df, use_container_width=True)

st.divider()

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Market Context")

    st.write(f"**Setup:** {setup}")

    if isinstance(invalidation, (int, float, np.floating)):
        st.write(f"**Invalidation level:** {invalidation:,.2f}")
    else:
        st.write(f"**Invalidation:** {invalidation}")

    st.write("**Regime reasons:**")
    for r in reasons:
        st.write(f"- {r}")

    st.write("**Day-type reasons:**")
    for r in day_type_reasons:
        st.write(f"- {r}")

with right:
    st.subheader("Risk Box")
    st.write(f"**Account size:** ${account_size:,.2f}")
    st.write(f"**Risk per trade:** ${risk_dollars:,.2f}")
    st.write(f"**Risk per contract:** ${risk_per_contract:,.2f}")
    st.write(f"**Max option premium deployed:** ${max_premium:,.2f}")
    st.write(f"**Daily max loss:** ${daily_max_loss:,.2f}")
    st.write(f"**Option stop price:** ${stop_price:,.2f}")
    st.write(f"**Option target price:** ${profit_target_price:,.2f}")

st.divider()

st.subheader("Live Levels and Filters")

levels = pd.DataFrame([
    {"Level": "Previous Day High", "Value": prev_high},
    {"Level": "Previous Day Low", "Value": prev_low},
    {"Level": "Previous Day Close", "Value": prev_close},
    {"Level": "Open Price", "Value": open_price},
    {"Level": "Gap %", "Value": gap_pct},
    {"Level": "Opening Range High", "Value": opening_high},
    {"Level": "Opening Range Low", "Value": opening_low},
    {"Level": "VWAP", "Value": vwap},
    {"Level": "9 EMA", "Value": ema9},
    {"Level": "20 EMA", "Value": ema20},
    {"Level": "20 EMA Slope", "Value": ema_slope},
    {"Level": "5-Min ATR", "Value": atr_5m},
    {"Level": "RVOL", "Value": rvol},
    {"Level": "Breadth Score", "Value": breadth},
    {"Level": "Breadth Persistence Count", "Value": breadth_count},
])

st.dataframe(levels, use_container_width=True)

if breadth_mode == "Auto ETF Breadth Proxy" and not breadth_df.empty:
    st.subheader("ETF Breadth Components")
    st.dataframe(breadth_df, use_container_width=True)

if not breadth_history_df.empty:
    with st.expander("Breadth Persistence History", expanded=False):
        st.dataframe(breadth_history_df, use_container_width=True)

if not manual_market_override and not price_df.empty:
    st.subheader("Intraday SPY Chart")

    chart_df = price_df.set_index("datetime")[["Close"]].copy()
    chart_df["VWAP"] = vwap
    chart_df["Opening High"] = opening_high
    chart_df["Opening Low"] = opening_low

    st.line_chart(chart_df, use_container_width=True)

st.divider()

st.subheader("Trade Journal Entry")

with st.form("journal_form"):
    j1, j2, j3 = st.columns(3)

    with j1:
        trade_type = st.selectbox("Trade Type", ["CALL", "PUT", "No Trade", "Watch Only"])
        entry_price = st.number_input("Entry Option Price", min_value=0.0, value=option_price, step=0.05)
        exit_price = st.number_input("Exit Option Price", min_value=0.0, value=0.0, step=0.05)

    with j2:
        contracts_taken = st.number_input("Contracts Taken", min_value=0, value=max(contracts, 0), step=1)
        entry_underlying = st.number_input("Entry Underlying Price", value=price, step=0.1)
        exit_underlying = st.number_input("Exit Underlying Price", value=price, step=0.1)

    with j3:
        notes = st.text_area("Notes", placeholder="What did you see? Did you follow the plan?")
        followed_plan = st.checkbox("Followed plan?", value=True)

    submitted = st.form_submit_button("Generate Journal Row")

if submitted:
    pnl = (exit_price - entry_price) * contracts_taken * 100

    journal_row = pd.DataFrame([{
        "timestamp": datetime.now(EASTERN).strftime("%Y-%m-%d %H:%M:%S"),
        "trade_type": trade_type,
        "regime": regime,
        "regime_score": score,
        "signal": signal,
        "decision": decision,
        "decision_reason": decision_reason,
        "day_type": day_type,
        "rvol": rvol,
        "gap_type": gap_type,
        "gap_pct": gap_pct,
        "breadth": breadth,
        "breadth_source": breadth_source,
        "breadth_persistence": breadth_count,
        "entry_underlying": entry_underlying,
        "exit_underlying": exit_underlying,
        "entry_option": entry_price,
        "exit_option": exit_price,
        "contracts": contracts_taken,
        "pnl": pnl,
        "followed_plan": followed_plan,
        "notes": notes
    }])

    st.dataframe(journal_row, use_container_width=True)

    csv = journal_row.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Journal Row as CSV",
        data=csv,
        file_name="spx_0dte_gated_journal_row.csv",
        mime="text/csv"
    )

# ============================================================
# PLACE THIS AFTER decision, decision_reason, gate_df ARE CREATED
# ============================================================

if "previous_state" not in st.session_state:
    st.session_state.previous_state = {
        "decision": None,
        "signal": None,
        "regime": None,
        "day_type": None
    }

current_state = {
    "decision": decision,
    "signal": signal,
    "regime": regime,
    "day_type": day_type,
    "gate_status": tuple(gate_df["Status"].tolist())
}

previous_state = st.session_state.previous_state

state_changed = current_state != previous_state

if state_changed:
    log_event(
        timestamp=now_et,
        price=price,
        regime=regime,
        score=score,
        day_type=day_type,
        breadth=breadth,
        breadth_count=breadth_count,
        rvol=rvol,
        signal=signal,
        decision=decision,
        decision_reason=decision_reason,
        gate_df=gate_df
    )

    st.session_state.previous_state = current_state



# ============================================================
# PLACE THIS NEAR THE BOTTOM OF THE APP
# ============================================================

st.divider()

st.subheader("Automatic Event Log")

if os.path.exists(LOG_FILE):

    log_df = pd.read_csv(LOG_FILE)

    st.dataframe(
        log_df.tail(50),
        use_container_width=True
    )

    csv = log_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Full Event Log",
        data=csv,
        file_name="spx_event_log.csv",
        mime="text/csv"
    )

else:
    st.info("No events logged yet.")
