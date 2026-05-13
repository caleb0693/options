
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
# App Configuration
# ============================================================

st.set_page_config(
    page_title="SPX 0DTE Decision Dashboard",
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
# Market Data Functions - yfinance only
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
def get_daily_data(symbol: str, period: str = "10d") -> pd.DataFrame:
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


def calculate_vwap(df: pd.DataFrame) -> float:
    if df.empty or "Volume" not in df.columns or df["Volume"].sum() == 0:
        return np.nan

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    return float((typical_price * df["Volume"]).sum() / df["Volume"].sum())


def calculate_ema(df: pd.DataFrame, span: int) -> float:
    if df.empty or "Close" not in df.columns:
        return np.nan

    return float(df["Close"].ewm(span=span, adjust=False).mean().iloc[-1])


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
# Automatic ETF Breadth Proxy
# ============================================================

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

        rows.append(
            {
                "Symbol": symbol,
                "Last": round(last_price, 2),
                "VWAP": round(vwap, 2),
                "Above VWAP": above_vwap,
                "% From VWAP": round(pct_from_vwap, 2)
            }
        )

    total = positive + negative

    if total == 0:
        breadth = 0.0
    else:
        breadth = (positive - negative) / total

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
# Decision Logic
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


def final_trade_decision(signal, score, checklist, contracts):
    failed_rules = [rule for rule, passed in checklist.items() if not passed]

    if signal == "CALL Watch" and score >= 3 and all(checklist.values()) and contracts > 0:
        return "BUY CALL", "Bullish regime, breakout confirmation, checklist passed, and position size is within risk limits."

    if signal == "PUT Watch" and score <= -3 and all(checklist.values()) and contracts > 0:
        return "BUY PUT", "Bearish regime, breakdown confirmation, checklist passed, and position size is within risk limits."

    if failed_rules:
        return "STAY OUT", "Trade blocked because: " + "; ".join(failed_rules[:4])

    return "STAY OUT", "Trade blocked because the directional setup is not strong enough."


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("SPX 0DTE Controls")

refresh_seconds = st.sidebar.selectbox(
    "Refresh interval",
    [15, 30, 60, 120],
    index=1
)

if st_autorefresh is not None:
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
    # SPY is used as a liquid proxy for SPX movement.
    underlying_symbol = "SPY"
    vix_symbol = "^VIX"

    price_df = get_intraday_data(underlying_symbol)
    vix_df = get_intraday_data(vix_symbol)

    if price_df.empty:
        st.error("Could not load SPY data from yfinance. Try manual market override.")
        st.stop()

    price = float(price_df["Close"].iloc[-1])
    last_timestamp = price_df["datetime"].iloc[-1]

    vix = float(vix_df["Close"].iloc[-1]) if not vix_df.empty else np.nan

    prev_high, prev_low, prev_close = get_previous_day_levels(underlying_symbol)
    opening_high, opening_low = get_opening_range(price_df)

    vwap = calculate_vwap(price_df)
    ema9 = calculate_ema(price_df, 9)
    ema20 = calculate_ema(price_df, 20)

    price_5m_df = get_intraday_data(underlying_symbol, interval="5m")
    atr_5m = calculate_atr(price_5m_df, 14)

    current_time = now_et.time()

else:
    st.sidebar.subheader("Manual Market Inputs")

    price = st.sidebar.number_input("Current Price / SPY Proxy", value=530.0, step=0.1)
    prev_high = st.sidebar.number_input("Previous Day High", value=532.0, step=0.1)
    prev_low = st.sidebar.number_input("Previous Day Low", value=526.5, step=0.1)
    opening_high = st.sidebar.number_input("Opening Range High", value=531.0, step=0.1)
    opening_low = st.sidebar.number_input("Opening Range Low", value=528.5, step=0.1)
    vwap = st.sidebar.number_input("VWAP", value=529.8, step=0.1)
    ema9 = st.sidebar.number_input("9 EMA", value=530.5, step=0.1)
    ema20 = st.sidebar.number_input("20 EMA", value=529.5, step=0.1)
    atr_5m = st.sidebar.number_input("5-Min ATR", value=0.6, step=0.1)
    vix = st.sidebar.number_input("VIX", value=16.0, step=0.5)
    current_time = st.sidebar.time_input("Market Time", value=time(9, 45))
    last_timestamp = now_et
    price_df = pd.DataFrame()


# ============================================================
# Main App
# ============================================================

st.title("SPX 0DTE Decision Dashboard")
st.caption("Uses SPY as a liquid proxy for SPX directional logic. Decision support only. Does not execute trades.")

with st.expander("How Auto ETF Breadth Proxy Works", expanded=False):
    st.markdown("""
    The app checks whether major liquid ETFs and sector ETFs are above or below their intraday VWAP.

    **Symbols used:** SPY, QQQ, IWM, XLK, XLF, XLV, XLY, XLI, XLE, XLP, XLU, XLB, XLRE, XLC

    **Formula:**

    `breadth_score = (ETFs above VWAP - ETFs below VWAP) / total ETFs`

    **Interpretation:**

    - `+0.60 to +1.00` = strong bullish participation
    - `-0.60 to -1.00` = strong bearish participation
    - `-0.59 to +0.59` = mixed / choppy participation
    """)

with st.expander("Entry Decision Guide", expanded=False):
    st.markdown("""
    ### Final Command Logic

    - **BUY CALL**: bullish regime, price above VWAP, breakout above opening range high, checklist passed.
    - **BUY PUT**: bearish regime, price below VWAP, breakdown below opening range low, checklist passed.
    - **STAY OUT**: mixed, late, unclear, oversized, or incomplete setup.

    ### CALL Entry Rules

    - Regime score ≥ +3
    - Signal = CALL Watch
    - Price above VWAP
    - Price above opening range high
    - Strong positive breadth
    - Checklist passed
    - Contract count greater than zero

    ### PUT Entry Rules

    - Regime score ≤ -3
    - Signal = PUT Watch
    - Price below VWAP
    - Price below opening range low
    - Strong negative breadth
    - Checklist passed
    - Contract count greater than zero
    """)

top1, top2, top3, top4 = st.columns(4)

top1.metric("SPY Proxy", f"{price:,.2f}")
top2.metric("VIX", "N/A" if pd.isna(vix) else f"{vix:,.2f}")
top3.metric("Breadth", f"{breadth:.2f}")
top4.metric("Last Bar", str(last_timestamp).split(".")[0])

st.caption(f"Breadth Source: {breadth_source}")

if breadth_mode == "Auto ETF Breadth Proxy":
    st.caption(f"ETF Breadth: {positive} above VWAP / {negative} below VWAP")

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

st.divider()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Regime", regime)
c2.metric("Regime Score", score)
c3.metric("Signal", signal)
c4.metric("Suggested Contracts", contracts)

st.divider()

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Trade Decision")

    if signal == "No Trade":
        st.warning("No-trade condition based on current filters.")
    elif signal == "CALL Watch":
        st.success("CALL setup is active, pending execution discipline.")
    elif signal == "PUT Watch":
        st.error("PUT setup is active, pending execution discipline.")

    st.write(f"**Setup:** {setup}")

    if isinstance(invalidation, (int, float, np.floating)):
        st.write(f"**Invalidation level:** {invalidation:,.2f}")
    else:
        st.write(f"**Invalidation:** {invalidation}")

    st.write("**Regime reasons:**")
    for r in reasons:
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

st.subheader("Live Levels")

levels = pd.DataFrame([
    {"Level": "Previous Day High", "Value": prev_high},
    {"Level": "Previous Day Low", "Value": prev_low},
    {"Level": "Opening Range High", "Value": opening_high},
    {"Level": "Opening Range Low", "Value": opening_low},
    {"Level": "VWAP", "Value": vwap},
    {"Level": "9 EMA", "Value": ema9},
    {"Level": "20 EMA", "Value": ema20},
    {"Level": "5-Min ATR", "Value": atr_5m},
    {"Level": "Breadth Score", "Value": breadth},
])

st.dataframe(levels, use_container_width=True)

if breadth_mode == "Auto ETF Breadth Proxy" and not breadth_df.empty:
    st.subheader("ETF Breadth Components")
    st.dataframe(breadth_df, use_container_width=True)

if not manual_market_override and not price_df.empty:
    st.subheader("Intraday SPY Chart")

    chart_df = price_df.set_index("datetime")[["Close"]].copy()
    chart_df["VWAP"] = vwap
    chart_df["Opening High"] = opening_high
    chart_df["Opening Low"] = opening_low

    st.line_chart(chart_df, use_container_width=True)

st.divider()

st.subheader("Execution Checklist")

checklist = {
    "Inside preferred trading window": time(9, 35) <= current_time <= time(10, 30) or time(13, 30) <= current_time <= time(15, 0),
    "Clear directional regime": abs(score) >= 3,
    "Price aligned with VWAP": (price > vwap and signal == "CALL Watch") or (price < vwap and signal == "PUT Watch"),
    "Defined invalidation level": isinstance(invalidation, (int, float, np.floating)),
    "Contracts within risk limit": contracts > 0,
    "No averaging down": True,
    "Take partial profit at target": True,
}

check_df = pd.DataFrame(
    [{"Rule": k, "Pass": "Yes" if v else "No"} for k, v in checklist.items()]
)

st.dataframe(check_df, use_container_width=True)

if all(checklist.values()) and signal != "No Trade":
    st.success("Checklist passed. Trade is eligible under this model.")
else:
    st.info("Checklist not fully passed. Wait, reduce size, or skip.")

decision, decision_reason = final_trade_decision(
    signal=signal,
    score=score,
    checklist=checklist,
    contracts=contracts
)

st.divider()

st.subheader("Final Trade Decision")

if decision == "BUY CALL":
    st.success(f"### {decision}")
elif decision == "BUY PUT":
    st.error(f"### {decision}")
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

st.subheader("Trade Journal Entry")

with st.form("journal_form"):
    j1, j2, j3 = st.columns(3)

    with j1:
        trade_type = st.selectbox("Trade Type", ["CALL", "PUT", "No Trade"])
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
        "signal": signal,
        "decision": decision,
        "entry_underlying": entry_underlying,
        "exit_underlying": exit_underlying,
        "entry_option": entry_price,
        "exit_option": exit_price,
        "contracts": contracts_taken,
        "pnl": pnl,
        "breadth": breadth,
        "breadth_source": breadth_source,
        "followed_plan": followed_plan,
        "notes": notes
    }])

    st.dataframe(journal_row, use_container_width=True)

    csv = journal_row.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Journal Row as CSV",
        data=csv,
        file_name="spx_0dte_journal_row.csv",
        mime="text/csv"
    )
