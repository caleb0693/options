# ============================================================
# Live 0DTE SPX App V2
# Signal Quality + Regime + Execution Gate Architecture
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, time, date
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ============================================================
# Streamlit Config
# ============================================================

st.set_page_config(
    page_title="Live 0DTE SPX App V2",
    layout="wide"
)

st.title("Live 0DTE SPX App V2")
st.caption("Signal-quality, regime-aware, gated decision-support tool for 0DTE SPX-style trading.")

st.warning(
    "This app uses SPY or QQQ intraday market data as a live proxy. "
    "It does not model real option premiums, theta, gamma, IV, slippage, or bid/ask spread."
)


# ============================================================
# Manual Refresh Controls
# ============================================================

refresh_col1, refresh_col2, refresh_col3 = st.columns([1, 2, 6])

with refresh_col1:
    if st.button("🔄 Refresh"):
        st.rerun()

with refresh_col2:
    st.metric(
        "Last Refresh",
        datetime.now().strftime("%H:%M:%S")
    )

with refresh_col3:
    st.caption(
        "Manual refresh enabled. "
        "Click refresh to reload live market data and recalculate signals."
    )



# ============================================================
# File Paths
# ============================================================

LOG_FILE = Path("0dte_live_v2_log.csv")
TRADE_FILE = Path("0dte_live_v2_trades.csv")


# ============================================================
# Utility Functions
# ============================================================

def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df



def load_live_data(symbol="SPY", period="5d", interval="5m"):
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    df = flatten_columns(df)

    if df.empty:
        return pd.DataFrame()

    df = df.reset_index()

    datetime_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={datetime_col: "datetime"})

    df["datetime"] = pd.to_datetime(df["datetime"])

    # Convert timezone safely to US/Eastern
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    else:
        df["datetime"] = df["datetime"].dt.tz_convert("America/New_York")

    df["datetime"] = df["datetime"].dt.tz_localize(None)

    df["date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.time

    # Regular market hours in Eastern time
    df = df[
        (df["time"] >= time(9, 30)) &
        (df["time"] <= time(16, 0))
    ].copy()

    return df

def calculate_vwap(df):
    df = df.copy()

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    df["pv"] = typical_price * df["Volume"]

    df["cum_pv"] = df.groupby("date")["pv"].cumsum()
    df["cum_vol"] = df.groupby("date")["Volume"].cumsum()

    df["vwap"] = df["cum_pv"] / df["cum_vol"]

    return df


def calculate_opening_range(df, opening_minutes=30):
    df = df.copy()
    df["or_high"] = np.nan
    df["or_low"] = np.nan

    for d, group in df.groupby("date"):
        day_start = group["datetime"].min()
        or_end = day_start + pd.Timedelta(minutes=opening_minutes)

        or_window = group[group["datetime"] < or_end]

        if or_window.empty:
            continue

        df.loc[df["date"] == d, "or_high"] = or_window["High"].max()
        df.loc[df["date"] == d, "or_low"] = or_window["Low"].min()

    return df


def calculate_indicators(df):
    df = df.copy()

    df = calculate_vwap(df)
    df = calculate_opening_range(df)

    df["ema_9"] = df.groupby("date")["Close"].transform(
        lambda x: x.ewm(span=9, adjust=False).mean()
    )

    df["ema_21"] = df.groupby("date")["Close"].transform(
        lambda x: x.ewm(span=21, adjust=False).mean()
    )

    df["bar_return"] = df.groupby("date")["Close"].pct_change()

    df["volume_ma"] = df.groupby("date")["Volume"].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )

    df["rvol"] = df["Volume"] / df["volume_ma"]

    df["above_vwap"] = df["Close"] > df["vwap"]
    df["below_vwap"] = df["Close"] < df["vwap"]

    df["above_or"] = df["Close"] > df["or_high"]
    df["below_or"] = df["Close"] < df["or_low"]

    df["trend_up"] = (df["ema_9"] > df["ema_21"]) & (df["Close"] > df["vwap"])
    df["trend_down"] = (df["ema_9"] < df["ema_21"]) & (df["Close"] < df["vwap"])

    return df


def classify_regime(row):
    if row["trend_up"] and row["above_or"]:
        return "TREND_UP"

    if row["trend_down"] and row["below_or"]:
        return "TREND_DOWN"

    if row["Close"] > row["vwap"] and row["ema_9"] > row["ema_21"]:
        return "BULLISH"

    if row["Close"] < row["vwap"] and row["ema_9"] < row["ema_21"]:
        return "BEARISH"

    return "CHOP"


def score_signal(row):
    call_score = 0
    put_score = 0

    call_reasons = []
    put_reasons = []

    # CALL score
    if row["Close"] > row["vwap"]:
        call_score += 1
        call_reasons.append("Price above VWAP")

    if row["ema_9"] > row["ema_21"]:
        call_score += 1
        call_reasons.append("EMA9 above EMA21")

    if row["Close"] > row["or_high"]:
        call_score += 2
        call_reasons.append("Break above opening range")

    if row["rvol"] >= 1.2:
        call_score += 1
        call_reasons.append("Elevated RVOL")

    if row["bar_return"] > 0:
        call_score += 1
        call_reasons.append("Positive bar momentum")

    # PUT score
    if row["Close"] < row["vwap"]:
        put_score += 1
        put_reasons.append("Price below VWAP")

    if row["ema_9"] < row["ema_21"]:
        put_score += 1
        put_reasons.append("EMA9 below EMA21")

    if row["Close"] < row["or_low"]:
        put_score += 2
        put_reasons.append("Break below opening range")

    if row["rvol"] >= 1.2:
        put_score += 1
        put_reasons.append("Elevated RVOL")

    if row["bar_return"] < 0:
        put_score += 1
        put_reasons.append("Negative bar momentum")

    if call_score > put_score and call_score >= 4:
        direction = "CALL"
        score = call_score
        reasons = call_reasons

    elif put_score > call_score and put_score >= 4:
        direction = "PUT"
        score = put_score
        reasons = put_reasons

    else:
        direction = "NONE"
        score = max(call_score, put_score)
        reasons = ["No dominant directional setup"]

    if score >= 6:
        quality = "A+"
    elif score == 5:
        quality = "A"
    elif score == 4:
        quality = "B"
    elif score == 3:
        quality = "C"
    else:
        quality = "D"

    return direction, score, quality, reasons, call_score, put_score


def quality_passes(quality, min_quality):
    rank = {"D": 0, "C": 1, "B": 2, "A": 3, "A+": 4}
    return rank[quality] >= rank[min_quality]


def load_trade_state():
    if TRADE_FILE.exists():
        trades = pd.read_csv(TRADE_FILE)
    else:
        trades = pd.DataFrame(columns=[
            "timestamp",
            "date",
            "direction",
            "signal_quality",
            "regime",
            "entry_price",
            "planned_stop",
            "planned_target",
            "status",
            "outcome_r"
        ])

    return trades


def save_trade_record(record):
    trades = load_trade_state()
    trades = pd.concat([trades, pd.DataFrame([record])], ignore_index=True)
    trades.to_csv(TRADE_FILE, index=False)


def log_event(record):
    if LOG_FILE.exists():
        log_df = pd.read_csv(LOG_FILE)
    else:
        log_df = pd.DataFrame()

    log_df = pd.concat([log_df, pd.DataFrame([record])], ignore_index=True)
    log_df.to_csv(LOG_FILE, index=False)


def evaluate_execution_gate(
    row,
    direction,
    quality,
    min_quality,
    regime,
    trades_today,
    max_trades_per_day,
    max_consecutive_losses,
    cooldown_minutes,
    allow_chop,
    start_trade_time,
    end_trade_time
):
    reasons = []

    now_time = row["time"]

    if now_time < start_trade_time:
        reasons.append("Before allowed trading window")

    if now_time > end_trade_time:
        reasons.append("After allowed trading window")

    if direction == "NONE":
        reasons.append("No directional signal")

    if not quality_passes(quality, min_quality):
        reasons.append(f"Signal quality below minimum threshold: {quality}")

    if regime == "CHOP" and not allow_chop:
        reasons.append("CHOP regime blocked")

    if len(trades_today) >= max_trades_per_day:
        reasons.append("Max trades per day reached")

    if not trades_today.empty:
        completed = trades_today.dropna(subset=["outcome_r"]).copy()

        if not completed.empty:
            completed["outcome_r"] = pd.to_numeric(completed["outcome_r"], errors="coerce")

            last_losses = completed.tail(max_consecutive_losses)

            if len(last_losses) >= max_consecutive_losses and (last_losses["outcome_r"] < 0).all():
                reasons.append("Consecutive loss lockout active")

        last_trade_time = pd.to_datetime(trades_today["timestamp"]).max()
        current_time = pd.to_datetime(row["datetime"])

        minutes_since_last_trade = (current_time - last_trade_time).total_seconds() / 60

        if minutes_since_last_trade < cooldown_minutes:
            reasons.append(f"Cooldown active: {minutes_since_last_trade:.0f} min since last trade")

    trade_allowed = len(reasons) == 0

    if trade_allowed:
        reasons.append("All execution gates passed")

    return trade_allowed, reasons


def calculate_trade_plan(row, direction, stop_points, target_points):
    entry = row["Close"]

    if direction == "CALL":
        stop = entry - stop_points
        target = entry + target_points

    elif direction == "PUT":
        stop = entry + stop_points
        target = entry - target_points

    else:
        stop = np.nan
        target = np.nan

    return entry, stop, target


def format_reasons(reasons):
    return "; ".join(reasons)


# ============================================================
# Sidebar Settings
# ============================================================

with st.sidebar:
    st.header("Market Data")

    symbol = st.selectbox(
        "Proxy Symbol",
        ["SPY", "QQQ"],
        index=0
    )

    period = st.selectbox(
        "Lookback Period",
        ["1d", "5d"],
        index=1
    )

    interval = st.selectbox(
        "Candle Interval",
        ["1m", "2m", "5m", "15m"],
        index=2
    )

    st.header("Signal Settings")

    min_quality = st.selectbox(
        "Minimum Trade Quality",
        ["A+", "A", "B", "C"],
        index=2
    )

    allow_chop = st.checkbox(
        "Allow CHOP Regime Trades",
        value=False
    )

    st.header("Trading Window")

    start_trade_time = st.time_input(
        "Start Trading",
        value=time(09, 30)
    )

    end_trade_time = st.time_input(
        "End Trading",
        value=time(15, 45)
    )

    st.header("Risk Controls")

    stop_points = st.number_input(
        "Stop Points",
        min_value=0.05,
        value=0.50,
        step=0.05
    )

    target_points = st.number_input(
        "Target Points",
        min_value=0.05,
        value=0.50,
        step=0.05
    )

    max_trades_per_day = st.number_input(
        "Max Trades Per Day",
        min_value=1,
        value=3,
        step=1
    )

    max_consecutive_losses = st.number_input(
        "Max Consecutive Losses",
        min_value=1,
        value=2,
        step=1
    )

    cooldown_minutes = st.number_input(
        "Cooldown Minutes",
        min_value=0,
        value=15,
        step=5
    )


# ============================================================
# Data Load
# ============================================================

df = load_live_data(symbol=symbol, period=period, interval=interval)

if df.empty:
    st.error("No market data loaded.")
    st.stop()

df = calculate_indicators(df)

df["regime"] = df.apply(classify_regime, axis=1)

signal_results = df.apply(score_signal, axis=1)
df["signal_direction"] = [x[0] for x in signal_results]
df["signal_score"] = [x[1] for x in signal_results]
df["signal_quality"] = [x[2] for x in signal_results]
df["signal_reasons"] = [format_reasons(x[3]) for x in signal_results]
df["call_score"] = [x[4] for x in signal_results]
df["put_score"] = [x[5] for x in signal_results]

latest = df.iloc[-1]

today = latest["date"]

trades = load_trade_state()

if not trades.empty and "date" in trades.columns:
    trades_today = trades[trades["date"].astype(str) == str(today)].copy()
else:
    trades_today = pd.DataFrame()

direction = latest["signal_direction"]
quality = latest["signal_quality"]
regime = latest["regime"]

trade_allowed, gate_reasons = evaluate_execution_gate(
    row=latest,
    direction=direction,
    quality=quality,
    min_quality=min_quality,
    regime=regime,
    trades_today=trades_today,
    max_trades_per_day=max_trades_per_day,
    max_consecutive_losses=max_consecutive_losses,
    cooldown_minutes=cooldown_minutes,
    allow_chop=allow_chop,
    start_trade_time=start_trade_time,
    end_trade_time=end_trade_time
)

entry, planned_stop, planned_target = calculate_trade_plan(
    latest,
    direction,
    stop_points,
    target_points
)

# ============================================================
# Main Status Display
# ============================================================

st.subheader("Current Signal State")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Proxy", symbol)
c2.metric("Last Price", f"{latest['Close']:.2f}")
c3.metric("Regime", regime)
c4.metric("Direction", direction)
c5.metric("Quality", quality)

c6, c7, c8, c9 = st.columns(4)

c6.metric("Signal Score", int(latest["signal_score"]))
c7.metric("CALL Score", int(latest["call_score"]))
c8.metric("PUT Score", int(latest["put_score"]))
c9.metric("RVOL", f"{latest['rvol']:.2f}" if pd.notna(latest["rvol"]) else "N/A")

st.divider()

if trade_allowed:
    st.success("TRADE ALLOWED")
else:
    st.error("NO TRADE / TRADE BLOCKED")

st.write("**Gate Status:**")
for reason in gate_reasons:
    st.write(f"- {reason}")

st.write("**Signal Reasons:**")
for reason in str(latest["signal_reasons"]).split(";"):
    st.write(f"- {reason.strip()}")

# ============================================================
# Trade Plan
# ============================================================

st.subheader("Trade Plan")

p1, p2, p3, p4 = st.columns(4)

p1.metric("Entry Reference", f"{entry:.2f}" if pd.notna(entry) else "N/A")
p2.metric("Planned Stop", f"{planned_stop:.2f}" if pd.notna(planned_stop) else "N/A")
p3.metric("Planned Target", f"{planned_target:.2f}" if pd.notna(planned_target) else "N/A")

rr = target_points / stop_points if stop_points > 0 else np.nan
p4.metric("Reward/Risk", f"{rr:.2f}R")

# ============================================================
# Manual Trade Logging
# ============================================================

st.subheader("Manual Trade Logging")

st.caption(
    "Use this to record when you actually take a trade. "
    "Later, enter the outcome in R manually or edit the CSV."
)

with st.form("trade_log_form"):
    actual_trade = st.checkbox("I took this trade")
    manual_outcome_r = st.number_input(
        "Outcome R, optional. Example: +1, -1, +0.5",
        value=0.0,
        step=0.25
    )
    include_outcome_now = st.checkbox("Include outcome now", value=False)

    submitted = st.form_submit_button("Save Trade Record")

    if submitted:
        if actual_trade:
            record = {
                "timestamp": latest["datetime"],
                "date": latest["date"],
                "direction": direction,
                "signal_quality": quality,
                "regime": regime,
                "entry_price": entry,
                "planned_stop": planned_stop,
                "planned_target": planned_target,
                "status": "TAKEN",
                "outcome_r": manual_outcome_r if include_outcome_now else np.nan
            }

            save_trade_record(record)
            st.success("Trade record saved.")

        else:
            st.info("No trade record saved because 'I took this trade' was not checked.")

# ============================================================
# Event Logging
# ============================================================

event_record = {
    "timestamp": latest["datetime"],
    "date": latest["date"],
    "symbol": symbol,
    "price": latest["Close"],
    "regime": regime,
    "direction": direction,
    "signal_quality": quality,
    "signal_score": latest["signal_score"],
    "call_score": latest["call_score"],
    "put_score": latest["put_score"],
    "rvol": latest["rvol"],
    "trade_allowed": trade_allowed,
    "gate_reasons": format_reasons(gate_reasons),
    "signal_reasons": latest["signal_reasons"],
    "entry_reference": entry,
    "planned_stop": planned_stop,
    "planned_target": planned_target
}

if st.button("Log Current Signal Snapshot"):
    log_event(event_record)
    st.success("Signal snapshot logged.")

# ============================================================
# Chart
# ============================================================

st.subheader("Intraday Price Chart")

today_df = df[df["date"] == today].copy()

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=today_df["datetime"],
    open=today_df["Open"],
    high=today_df["High"],
    low=today_df["Low"],
    close=today_df["Close"],
    name="Price"
))

fig.add_trace(go.Scatter(
    x=today_df["datetime"],
    y=today_df["vwap"],
    mode="lines",
    name="VWAP"
))

fig.add_trace(go.Scatter(
    x=today_df["datetime"],
    y=today_df["ema_9"],
    mode="lines",
    name="EMA 9"
))

fig.add_trace(go.Scatter(
    x=today_df["datetime"],
    y=today_df["ema_21"],
    mode="lines",
    name="EMA 21"
))

fig.add_trace(go.Scatter(
    x=today_df["datetime"],
    y=today_df["or_high"],
    mode="lines",
    name="OR High"
))

fig.add_trace(go.Scatter(
    x=today_df["datetime"],
    y=today_df["or_low"],
    mode="lines",
    name="OR Low"
))

# ============================================================
# Signal Marker Overlay
# ============================================================

signal_plot_df = today_df.copy()

# Accepted directional signals based on current minimum quality
signal_plot_df["quality_pass"] = signal_plot_df["signal_quality"].apply(
    lambda q: quality_passes(q, min_quality)
)

call_signals = signal_plot_df[
    (signal_plot_df["signal_direction"] == "CALL") &
    (signal_plot_df["quality_pass"])
]

put_signals = signal_plot_df[
    (signal_plot_df["signal_direction"] == "PUT") &
    (signal_plot_df["quality_pass"])
]

rejected_signals = signal_plot_df[
    (signal_plot_df["signal_direction"].isin(["CALL", "PUT"])) &
    (~signal_plot_df["quality_pass"])
]

# CALL markers
fig.add_trace(go.Scatter(
    x=call_signals["datetime"],
    y=call_signals["Low"] * 0.999,
    mode="markers",
    name="CALL Signal",
    marker=dict(
        symbol="triangle-up",
        size=14,
        color="lime"
    ),
    text=[
        f"CALL<br>Quality: {q}<br>Score: {s}<br>Regime: {r}"
        for q, s, r in zip(
            call_signals["signal_quality"],
            call_signals["signal_score"],
            call_signals["regime"]
        )
    ],
    hoverinfo="text"
))

# PUT markers
fig.add_trace(go.Scatter(
    x=put_signals["datetime"],
    y=put_signals["High"] * 1.001,
    mode="markers",
    name="PUT Signal",
    marker=dict(
        symbol="triangle-down",
        size=14,
        color="red"
    ),
    text=[
        f"PUT<br>Quality: {q}<br>Score: {s}<br>Regime: {r}"
        for q, s, r in zip(
            put_signals["signal_quality"],
            put_signals["signal_score"],
            put_signals["regime"]
        )
    ],
    hoverinfo="text"
))

# Rejected signal markers
fig.add_trace(go.Scatter(
    x=rejected_signals["datetime"],
    y=rejected_signals["Close"],
    mode="markers",
    name="Rejected Signal",
    marker=dict(
        symbol="x",
        size=10
    ),
    text=[
        f"Rejected {d}<br>Quality: {q}<br>Score: {s}<br>Regime: {r}"
        for d, q, s, r in zip(
            rejected_signals["signal_direction"],
            rejected_signals["signal_quality"],
            rejected_signals["signal_score"],
            rejected_signals["regime"]
        )
    ],
    hoverinfo="text"
))


fig.update_layout(
    height=650,
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(
    range=[
        pd.Timestamp.combine(today, time(9, 30)),
        pd.Timestamp.combine(today, time(16, 0))
    ]
    ),
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Signal Table
# ============================================================

st.subheader("Recent Signal History")

cols = [
    "datetime",
    "Close",
    "vwap",
    "ema_9",
    "ema_21",
    "or_high",
    "or_low",
    "rvol",
    "regime",
    "signal_direction",
    "signal_quality",
    "signal_score",
    "call_score",
    "put_score",
    "signal_reasons"
]

st.dataframe(
    today_df[cols],
    use_container_width=True
)

# ============================================================
# Trade Records
# ============================================================

st.subheader("Today’s Trade Records")

if trades_today.empty:
    st.info("No trades recorded today.")
else:
    st.dataframe(trades_today, use_container_width=True)

# ============================================================
# Download Buttons
# ============================================================

st.subheader("Downloads")

if LOG_FILE.exists():
    with open(LOG_FILE, "rb") as f:
        st.download_button(
            "Download Signal Log CSV",
            data=f,
            file_name="0dte_live_v2_log.csv",
            mime="text/csv"
        )

if TRADE_FILE.exists():
    with open(TRADE_FILE, "rb") as f:
        st.download_button(
            "Download Trade Log CSV",
            data=f,
            file_name="0dte_live_v2_trades.csv",
            mime="text/csv"
        )

# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "V2 architecture: regime classification, signal quality grading, execution gates, cooldown logic, "
    "manual trade logging, and structured signal snapshots."
)
