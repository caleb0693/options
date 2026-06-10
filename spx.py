# ============================================================
# Live 0DTE SPX App V2.2
# Signal Quality + Regime + Execution Gate + 14 ETF Breadth + Reversal Risk
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, time
from zoneinfo import ZoneInfo
import plotly.graph_objects as go
from pathlib import Path

# ============================================================
# Streamlit Config
# ============================================================

st.set_page_config(
    page_title="Live 0DTE SPX App V2.2",
    layout="wide"
)

st.title("Live 0DTE SPX App V2.2")
st.caption("Signal-quality, regime-aware, breadth-confirmed, reversal-aware, gated decision-support tool for 0DTE SPX-style trading.")

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
    st.metric("Last Refresh", datetime.now(ZoneInfo("America/New_York")).strftime("%I:%M:%S %p ET"))

with refresh_col3:
    st.caption("Manual refresh enabled. Click refresh to reload market data and recalculate signals. Time shown in US/Eastern.")

# ============================================================
# File Paths
# ============================================================

LOG_FILE = Path("0dte_live_v2_2_log.csv")

# ============================================================
# Constants
# ============================================================

BREADTH_SYMBOLS = [
    "SPY", "QQQ", "DIA", "IWM", "VTI",
    "XLK", "SMH", "SOXX", "XLY", "XLI",
    "XLF", "XLE", "XLV", "XLP"
]

QUALITY_RANK = {"D": 0, "C": 1, "B": 2, "A": 3, "A+": 4}
QUALITY_BY_RANK = {0: "D", 1: "C", 2: "B", 3: "A", 4: "A+"}

# ============================================================
# Utility Functions
# ============================================================

def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df


def standardize_intraday_df(df):
    df = flatten_columns(df)

    if df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    datetime_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={datetime_col: "datetime"})

    df["datetime"] = pd.to_datetime(df["datetime"])

    # Convert timezone safely to US/Eastern.
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    else:
        df["datetime"] = df["datetime"].dt.tz_convert("America/New_York")

    df["datetime"] = df["datetime"].dt.tz_localize(None)
    df["date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.time

    # Regular market hours in Eastern time.
    df = df[
        (df["time"] >= time(9, 30)) &
        (df["time"] <= time(16, 0))
    ].copy()

    return df


def load_live_data(symbol="SPY", period="5d", interval="5m"):
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False
    )
    return standardize_intraday_df(df)


def calculate_vwap(df):
    df = df.copy()
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    df["pv"] = typical_price * df["Volume"]
    df["cum_pv"] = df.groupby("date")["pv"].cumsum()
    df["cum_vol"] = df.groupby("date")["Volume"].cumsum()
    df["vwap"] = np.where(df["cum_vol"] > 0, df["cum_pv"] / df["cum_vol"], np.nan)
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

    df["rvol"] = np.where(df["volume_ma"] > 0, df["Volume"] / df["volume_ma"], np.nan)

    df["above_vwap"] = df["Close"] > df["vwap"]
    df["below_vwap"] = df["Close"] < df["vwap"]
    df["above_or"] = df["Close"] > df["or_high"]
    df["below_or"] = df["Close"] < df["or_low"]

    df["trend_up"] = (df["ema_9"] > df["ema_21"]) & (df["Close"] > df["vwap"])
    df["trend_down"] = (df["ema_9"] < df["ema_21"]) & (df["Close"] < df["vwap"])

    return df


def load_breadth_data(period="5d", interval="5m"):
    breadth_data = {}

    for symbol in BREADTH_SYMBOLS:
        try:
            raw = yf.download(
                symbol,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False
            )
            df = standardize_intraday_df(raw)
            if df.empty:
                continue
            df = calculate_indicators(df)
            breadth_data[symbol] = df
        except Exception:
            continue

    return breadth_data


def get_snapshot_at_or_before(df, current_time):
    if df.empty:
        return pd.Series(dtype="float64")

    subset = df[df["datetime"] <= current_time]
    if subset.empty:
        return pd.Series(dtype="float64")

    return subset.iloc[-1]


def calculate_breadth_alignment(breadth_data, current_time):
    rows = []
    bullish_count = 0
    bearish_count = 0
    neutral_count = 0

    for symbol, bdf in breadth_data.items():
        snap = get_snapshot_at_or_before(bdf, current_time)
        if snap.empty or pd.isna(snap.get("vwap", np.nan)):
            rows.append({
                "symbol": symbol,
                "close": np.nan,
                "vwap": np.nan,
                "state": "UNAVAILABLE",
                "rvol": np.nan
            })
            continue

        if snap["Close"] > snap["vwap"]:
            state = "BULLISH"
            bullish_count += 1
        elif snap["Close"] < snap["vwap"]:
            state = "BEARISH"
            bearish_count += 1
        else:
            state = "NEUTRAL"
            neutral_count += 1

        rows.append({
            "symbol": symbol,
            "close": snap["Close"],
            "vwap": snap["vwap"],
            "state": state,
            "rvol": snap.get("rvol", np.nan)
        })

    total_available = bullish_count + bearish_count + neutral_count

    if total_available == 0:
        breadth_state = "UNAVAILABLE"
    elif bullish_count >= 9:
        breadth_state = "STRONGLY BULLISH"
    elif bearish_count >= 9:
        breadth_state = "STRONGLY BEARISH"
    elif bullish_count >= 8:
        breadth_state = "BULLISH"
    elif bearish_count >= 8:
        breadth_state = "BEARISH"
    else:
        breadth_state = "MIXED"

    net_breadth_score = bullish_count - bearish_count

    breadth_df = pd.DataFrame(rows)

    return {
        "breadth_state": breadth_state,
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "neutral_count": neutral_count,
        "total_available": total_available,
        "net_breadth_score": net_breadth_score,
        "breadth_df": breadth_df
    }


def calculate_breadth_persistence(breadth_data, current_time, lookback_bars=3):
    states = []

    for symbol, bdf in breadth_data.items():
        subset = bdf[bdf["datetime"] <= current_time].tail(lookback_bars)
        if subset.empty or len(subset) < lookback_bars:
            continue

        above = (subset["Close"] > subset["vwap"]).all()
        below = (subset["Close"] < subset["vwap"]).all()

        if above:
            states.append("BULLISH")
        elif below:
            states.append("BEARISH")

    bullish_persistent = states.count("BULLISH")
    bearish_persistent = states.count("BEARISH")

    if bullish_persistent > bearish_persistent and bullish_persistent > 0:
        label = f"Bullish x{bullish_persistent}"
    elif bearish_persistent > bullish_persistent and bearish_persistent > 0:
        label = f"Bearish x{bearish_persistent}"
    else:
        label = "Mixed / Not Persistent"

    return label, bullish_persistent, bearish_persistent


def interpret_rvol(rvol):
    if pd.isna(rvol):
        return "Unavailable"
    if rvol <= 0:
        return "Zero / possible data issue or no participation"
    if rvol < 0.75:
        return "Weak volume"
    if rvol < 1.00:
        return "Slightly weak volume"
    if rvol < 1.25:
        return "Normal volume"
    if rvol < 1.75:
        return "Elevated volume"
    return "Strong volume"


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


def calculate_regime_score(row, breadth_info):
    score = 0

    if row["Close"] > row["vwap"]:
        score += 1
    elif row["Close"] < row["vwap"]:
        score -= 1

    if row["ema_9"] > row["ema_21"]:
        score += 1
    elif row["ema_9"] < row["ema_21"]:
        score -= 1

    if row["Close"] > row["or_high"]:
        score += 1
    elif row["Close"] < row["or_low"]:
        score -= 1

    breadth_state = breadth_info.get("breadth_state", "UNAVAILABLE")
    if breadth_state in ["BULLISH", "STRONGLY BULLISH"]:
        score += 1
    elif breadth_state in ["BEARISH", "STRONGLY BEARISH"]:
        score -= 1

    return int(score)


def regime_score_label(score):
    if score >= 3:
        return "Bullish trend / call-favored"
    if score == 2:
        return "Bullish leaning"
    if score <= -3:
        return "Bearish trend / put-favored"
    if score == -2:
        return "Bearish leaning"
    return "Neutral / mixed"


def classify_day_type(today_df, latest_row):
    if today_df.empty:
        return "Unavailable"

    day_range = today_df["High"].max() - today_df["Low"].min()
    current_distance_from_open = latest_row["Close"] - today_df.iloc[0]["Open"]
    vwap_crosses = ((today_df["Close"] > today_df["vwap"]).astype(int).diff().abs() == 1).sum()

    if latest_row["trend_up"] and latest_row["above_or"] and current_distance_from_open > 0:
        return "Bullish Trend Day"
    if latest_row["trend_down"] and latest_row["below_or"] and current_distance_from_open < 0:
        return "Bearish Trend Day"
    if vwap_crosses >= 4:
        return "Choppy / Mean-Reverting"
    if day_range <= 0:
        return "Neutral / Developing"
    return "Neutral / Developing"


def classify_gap_type(df, today):
    dates = sorted(df["date"].unique())
    if len(dates) < 2:
        return "Unavailable"

    today_df = df[df["date"] == today]
    prev_day = dates[dates.index(today) - 1] if today in dates and dates.index(today) > 0 else None

    if prev_day is None or today_df.empty:
        return "Unavailable"

    prev_df = df[df["date"] == prev_day]
    if prev_df.empty:
        return "Unavailable"

    today_open = today_df.iloc[0]["Open"]
    prev_high = prev_df["High"].max()
    prev_low = prev_df["Low"].min()
    prev_close = prev_df.iloc[-1]["Close"]

    if today_open > prev_high:
        return "Gap Above Prior Range"
    if today_open < prev_low:
        return "Gap Below Prior Range"
    if today_open > prev_close:
        return "Inside-Range Gap Up"
    if today_open < prev_close:
        return "Inside-Range Gap Down"
    return "Flat / No Gap"


def score_signal(row):
    call_score = 0
    put_score = 0
    call_reasons = []
    put_reasons = []

    rvol_valid = pd.notna(row.get("rvol", np.nan)) and row.get("rvol", np.nan) > 0

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
    if rvol_valid and row["rvol"] >= 1.2:
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
    if rvol_valid and row["rvol"] >= 1.2:
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

    if not rvol_valid:
        reasons.append("RVOL unavailable/zero; volume confirmation not awarded")

    return direction, score, quality, reasons, call_score, put_score


def adjust_quality_for_breadth(direction, quality, breadth_state):
    rank = QUALITY_RANK.get(quality, 0)
    original_quality = quality
    note = "Breadth neutral/no quality adjustment"

    bullish_breadth = breadth_state in ["BULLISH", "STRONGLY BULLISH"]
    bearish_breadth = breadth_state in ["BEARISH", "STRONGLY BEARISH"]

    if direction == "CALL" and bullish_breadth:
        rank = min(rank + 1, 4)
        note = "Breadth aligned bullish; quality upgraded"
    elif direction == "PUT" and bearish_breadth:
        rank = min(rank + 1, 4)
        note = "Breadth aligned bearish; quality upgraded"
    elif direction == "CALL" and bearish_breadth:
        rank = min(rank, 1)
        note = "Breadth disagrees bearish; quality downgraded"
    elif direction == "PUT" and bullish_breadth:
        rank = min(rank, 1)
        note = "Breadth disagrees bullish; quality downgraded"

    adjusted_quality = QUALITY_BY_RANK[rank]
    return adjusted_quality, note, original_quality


def quality_passes(quality, min_quality):
    return QUALITY_RANK[quality] >= QUALITY_RANK[min_quality]


def suggest_contracts(trade_allowed, quality, regime_score, rvol, max_contracts):
    if not trade_allowed:
        return 0

    base = 0
    if quality == "A+":
        base = max_contracts
    elif quality == "A":
        base = max(1, int(round(max_contracts * 0.75)))
    elif quality == "B":
        base = max(1, int(round(max_contracts * 0.50)))
    else:
        base = 0

    if abs(regime_score) < 2:
        base = min(base, 1)

    if pd.isna(rvol) or rvol <= 0 or rvol < 0.75:
        base = min(base, 1)

    return int(max(0, base))


def evaluate_execution_gate(
    row,
    direction,
    quality,
    min_quality,
    regime,
    allow_chop,
    start_trade_time,
    end_trade_time,
    breadth_state,
    require_breadth_alignment,
    reversal_state
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

    if require_breadth_alignment:
        if direction == "CALL" and breadth_state not in ["BULLISH", "STRONGLY BULLISH"]:
            reasons.append("Breadth not aligned bullish for CALL")
        if direction == "PUT" and breadth_state not in ["BEARISH", "STRONGLY BEARISH"]:
            reasons.append("Breadth not aligned bearish for PUT")

    # Reversal risk gate: do not allow continuation trades when a contrary reversal is elevated/confirmed.
    if direction == "PUT" and reversal_state in ["Bullish reversal risk elevated", "Bullish reversal confirmed"]:
        reasons.append("Bullish reversal risk conflicts with PUT continuation")
    if direction == "CALL" and reversal_state in ["Bearish reversal risk elevated", "Bearish reversal confirmed"]:
        reasons.append("Bearish reversal risk conflicts with CALL continuation")

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
    return "; ".join([str(r) for r in reasons])


def wick_ratio(row):
    body = abs(row["Close"] - row["Open"])
    body = max(body, 0.0001)
    upper_wick = row["High"] - max(row["Open"], row["Close"])
    lower_wick = min(row["Open"], row["Close"]) - row["Low"]
    return upper_wick / body, lower_wick / body


def calculate_reversal_risk(today_df, latest_row, breadth_state, breadth_persistence):
    """
    Symmetrical reversal-risk engine.

    This does not try to catch tops or bottoms. It detects when an existing trend is
    losing control and when the opposite side has started to confirm.
    """
    if today_df.empty or len(today_df) < 8:
        return {
            "reversal_state": "Insufficient data",
            "reversal_direction": "NONE",
            "reversal_score": 0,
            "trend_continuation_status": "Unknown",
            "reversal_reasons": ["Not enough intraday bars to evaluate reversal risk"]
        }

    recent_context = today_df.iloc[:-1].tail(12)
    if recent_context.empty:
        recent_context = today_df.tail(12)

    prior_bearish_count = (
        (recent_context["Close"] < recent_context["vwap"]) &
        (recent_context["ema_9"] < recent_context["ema_21"])
    ).sum()

    prior_bullish_count = (
        (recent_context["Close"] > recent_context["vwap"]) &
        (recent_context["ema_9"] > recent_context["ema_21"])
    ).sum()

    prior_trend = "NONE"
    if prior_bearish_count >= max(3, len(recent_context) // 2):
        prior_trend = "BEARISH"
    elif prior_bullish_count >= max(3, len(recent_context) // 2):
        prior_trend = "BULLISH"

    upper_wick_r, lower_wick_r = wick_ratio(latest_row)
    recent = today_df.tail(6)
    previous = today_df.iloc[:-1].tail(6)

    prev_low = previous["Low"].min() if not previous.empty else np.nan
    prev_high = previous["High"].max() if not previous.empty else np.nan

    bullish_score = 0
    bearish_score = 0
    bullish_reasons = []
    bearish_reasons = []

    # Bearish trend weakening into bullish reversal risk.
    if prior_trend == "BEARISH":
        if latest_row["Close"] > latest_row["ema_9"]:
            bullish_score += 1
            bullish_reasons.append("Price reclaimed EMA9 after bearish context")
        if latest_row["Close"] > latest_row["ema_21"]:
            bullish_score += 1
            bullish_reasons.append("Price reclaimed EMA21 after bearish context")
        if latest_row["Close"] > latest_row["vwap"]:
            bullish_score += 2
            bullish_reasons.append("Price reclaimed VWAP after bearish control")
        if pd.notna(prev_low) and latest_row["Low"] <= prev_low and latest_row["Close"] > latest_row["Open"]:
            bullish_score += 1
            bullish_reasons.append("Failed new low with bullish close")
        if lower_wick_r >= 1.5:
            bullish_score += 1
            bullish_reasons.append("Long lower wick suggests downside rejection")
        if breadth_state in ["MIXED", "BULLISH", "STRONGLY BULLISH"]:
            bullish_score += 1
            bullish_reasons.append("Breadth no longer strongly supports bearish continuation")
        if breadth_state in ["BULLISH", "STRONGLY BULLISH"]:
            bullish_score += 1
            bullish_reasons.append("Breadth flipped bullish against prior bearish trend")
        if not str(breadth_persistence).startswith("Bearish"):
            bullish_score += 1
            bullish_reasons.append("Bearish breadth persistence weakened")
        if latest_row.get("bar_return", 0) > 0 and pd.notna(latest_row.get("rvol", np.nan)) and latest_row.get("rvol", 0) >= 1.0:
            bullish_score += 1
            bullish_reasons.append("Positive candle with acceptable RVOL")

    # Bullish trend weakening into bearish reversal risk.
    if prior_trend == "BULLISH":
        if latest_row["Close"] < latest_row["ema_9"]:
            bearish_score += 1
            bearish_reasons.append("Price lost EMA9 after bullish context")
        if latest_row["Close"] < latest_row["ema_21"]:
            bearish_score += 1
            bearish_reasons.append("Price lost EMA21 after bullish context")
        if latest_row["Close"] < latest_row["vwap"]:
            bearish_score += 2
            bearish_reasons.append("Price lost VWAP after bullish control")
        if pd.notna(prev_high) and latest_row["High"] >= prev_high and latest_row["Close"] < latest_row["Open"]:
            bearish_score += 1
            bearish_reasons.append("Failed new high with bearish close")
        if upper_wick_r >= 1.5:
            bearish_score += 1
            bearish_reasons.append("Long upper wick suggests upside rejection")
        if breadth_state in ["MIXED", "BEARISH", "STRONGLY BEARISH"]:
            bearish_score += 1
            bearish_reasons.append("Breadth no longer strongly supports bullish continuation")
        if breadth_state in ["BEARISH", "STRONGLY BEARISH"]:
            bearish_score += 1
            bearish_reasons.append("Breadth flipped bearish against prior bullish trend")
        if not str(breadth_persistence).startswith("Bullish"):
            bearish_score += 1
            bearish_reasons.append("Bullish breadth persistence weakened")
        if latest_row.get("bar_return", 0) < 0 and pd.notna(latest_row.get("rvol", np.nan)) and latest_row.get("rvol", 0) >= 1.0:
            bearish_score += 1
            bearish_reasons.append("Negative candle with acceptable RVOL")

    # Classification.
    if bullish_score >= 6 and latest_row["Close"] > latest_row["vwap"] and breadth_state in ["BULLISH", "STRONGLY BULLISH"]:
        state = "Bullish reversal confirmed"
        direction = "BULLISH"
        score = bullish_score
        reasons = bullish_reasons
    elif bearish_score >= 6 and latest_row["Close"] < latest_row["vwap"] and breadth_state in ["BEARISH", "STRONGLY BEARISH"]:
        state = "Bearish reversal confirmed"
        direction = "BEARISH"
        score = bearish_score
        reasons = bearish_reasons
    elif bullish_score >= 3:
        state = "Bullish reversal risk elevated"
        direction = "BULLISH"
        score = bullish_score
        reasons = bullish_reasons
    elif bearish_score >= 3:
        state = "Bearish reversal risk elevated"
        direction = "BEARISH"
        score = bearish_score
        reasons = bearish_reasons
    elif prior_trend == "BEARISH":
        state = "Stable bearish trend"
        direction = "NONE"
        score = bullish_score
        reasons = bullish_reasons if bullish_reasons else ["Bearish continuation remains structurally intact"]
    elif prior_trend == "BULLISH":
        state = "Stable bullish trend"
        direction = "NONE"
        score = bearish_score
        reasons = bearish_reasons if bearish_reasons else ["Bullish continuation remains structurally intact"]
    else:
        state = "No clear reversal structure"
        direction = "NONE"
        score = max(bullish_score, bearish_score)
        reasons = ["No dominant prior trend or reversal structure detected"]

    if "risk elevated" in state:
        continuation_status = "Caution"
    elif "confirmed" in state:
        continuation_status = "Continuation invalidated"
    elif state.startswith("Stable"):
        continuation_status = "Continuation favored"
    else:
        continuation_status = "Neutral"

    return {
        "reversal_state": state,
        "reversal_direction": direction,
        "reversal_score": int(score),
        "trend_continuation_status": continuation_status,
        "reversal_reasons": reasons
    }

def build_full_day_signal_history(
    df,
    today,
    breadth_data,
    min_quality,
    allow_chop,
    require_breadth_alignment,
    start_trade_time,
    end_trade_time,
    stop_points,
    target_points,
    max_contracts,
    breadth_lookback_bars
):
    today_history = df[df["date"] == today].copy()

    records = []

    for i in range(len(today_history)):
        row = today_history.iloc[i]
        current_time = row["datetime"]

        partial_day = today_history.iloc[:i + 1].copy()

        b_info = calculate_breadth_alignment(breadth_data, current_time)
        b_state = b_info["breadth_state"]

        b_persistence, b_persistent_bull, b_persistent_bear = calculate_breadth_persistence(
            breadth_data,
            current_time,
            lookback_bars=int(breadth_lookback_bars)
        )

        raw_direction = row["signal_direction"]
        raw_quality = row["raw_signal_quality"]

        adjusted_quality, breadth_note, original_quality = adjust_quality_for_breadth(
            raw_direction,
            raw_quality,
            b_state
        )

        row_regime_score = calculate_regime_score(row, b_info)
        row_regime_description = regime_score_label(row_regime_score)
        row_day_type = classify_day_type(partial_day, row)
        row_gap_type = classify_gap_type(df, row["date"])
        row_rvol_label = interpret_rvol(row["rvol"])

        rev_info = calculate_reversal_risk(
            today_df=partial_day,
            latest_row=row,
            breadth_state=b_state,
            breadth_persistence=b_persistence
        )

        row_trade_allowed, row_gate_reasons = evaluate_execution_gate(
            row=row,
            direction=raw_direction,
            quality=adjusted_quality,
            min_quality=min_quality,
            regime=row["regime"],
            allow_chop=allow_chop,
            start_trade_time=start_trade_time,
            end_trade_time=end_trade_time,
            breadth_state=b_state,
            require_breadth_alignment=require_breadth_alignment,
            reversal_state=rev_info["reversal_state"]
        )

        row_suggested_contracts = suggest_contracts(
            row_trade_allowed,
            adjusted_quality,
            row_regime_score,
            row["rvol"],
            int(max_contracts)
        )

        row_final_signal = raw_direction if row_trade_allowed else "No Trade"

        entry_ref, planned_stop_ref, planned_target_ref = calculate_trade_plan(
            row,
            raw_direction,
            stop_points,
            target_points
        )

        records.append({
            "datetime": row["datetime"],
            "Close": row["Close"],
            "vwap": row["vwap"],
            "ema_9": row["ema_9"],
            "ema_21": row["ema_21"],
            "or_high": row["or_high"],
            "or_low": row["or_low"],
            "rvol": row["rvol"],
            "rvol_label": row_rvol_label,
            "regime": row["regime"],
            "regime_description": row_regime_description,
            "regime_score": row_regime_score,
            "signal_direction": raw_direction,
            "final_signal": row_final_signal,
            "raw_signal_quality": raw_quality,
            "signal_quality": adjusted_quality,
            "signal_score": row["signal_score"],
            "call_score": row["call_score"],
            "put_score": row["put_score"],
            "suggested_contracts": row_suggested_contracts,
            "day_type": row_day_type,
            "gap_type": row_gap_type,
            "breadth_state": b_state,
            "breadth_persistence": b_persistence,
            "breadth_bullish_count": b_info["bullish_count"],
            "breadth_bearish_count": b_info["bearish_count"],
            "net_breadth_score": b_info["net_breadth_score"],
            "breadth_total_available": b_info["total_available"],
            "reversal_risk": rev_info["reversal_state"],
            "reversal_direction": rev_info["reversal_direction"],
            "reversal_score": rev_info["reversal_score"],
            "trend_continuation": rev_info["trend_continuation_status"],
            "gate_reasons": format_reasons(row_gate_reasons),
            "signal_reasons": row["signal_reasons"],
            "breadth_quality_note": breadth_note,
            "entry_reference": entry_ref,
            "planned_stop": planned_stop_ref,
            "planned_target": planned_target_ref
        })

    return pd.DataFrame(records)

# ============================================================
# Sidebar Settings
# ============================================================

with st.sidebar:
    st.header("Market Data")

    symbol = st.selectbox("Proxy Symbol", ["SPY", "QQQ"], index=0)
    period = st.selectbox("Lookback Period", ["1d", "5d"], index=1)
    interval = st.selectbox("Candle Interval", ["1m", "2m", "5m", "15m"], index=2)

    st.header("Signal Settings")

    min_quality = st.selectbox("Minimum Trade Quality", ["A+", "A", "B", "C"], index=2)
    allow_chop = st.checkbox("Allow CHOP Regime Trades", value=False)
    require_breadth_alignment = st.checkbox("Require Breadth Alignment", value=False)

    st.header("Trading Window")

    start_trade_time = st.time_input("Start Trading", value=time(10, 00))
    end_trade_time = st.time_input("End Trading", value=time(15, 45))

    st.header("Risk Controls")

    stop_points = st.number_input("Stop Points", min_value=0.05, value=0.50, step=0.05)
    target_points = st.number_input("Target Points", min_value=0.05, value=0.50, step=0.05)
    max_contracts = st.number_input("Max Suggested Contracts", min_value=0, value=3, step=1)

    st.header("Breadth Settings")
    breadth_lookback_bars = st.number_input("Breadth Persistence Bars", min_value=2, value=3, step=1)

# ============================================================
# Data Load
# ============================================================

with st.spinner("Loading market and breadth data..."):
    df = load_live_data(symbol=symbol, period=period, interval=interval)

    if df.empty:
        st.error("No market data loaded.")
        st.stop()

    df = calculate_indicators(df)
    breadth_data = load_breadth_data(period=period, interval=interval)

# ============================================================
# Signal Calculations
# ============================================================

df["regime"] = df.apply(classify_regime, axis=1)

signal_results = df.apply(score_signal, axis=1)
df["signal_direction"] = [x[0] for x in signal_results]
df["signal_score"] = [x[1] for x in signal_results]
df["raw_signal_quality"] = [x[2] for x in signal_results]
df["signal_reasons"] = [format_reasons(x[3]) for x in signal_results]
df["call_score"] = [x[4] for x in signal_results]
df["put_score"] = [x[5] for x in signal_results]
df["rvol_label"] = df["rvol"].apply(interpret_rvol)

latest = df.iloc[-1]
today = latest["date"]
today_df = df[df["date"] == today].copy()

breadth_info = calculate_breadth_alignment(breadth_data, latest["datetime"])
breadth_state = breadth_info["breadth_state"]
breadth_df = breadth_info["breadth_df"]
breadth_persistence, breadth_persistent_bull, breadth_persistent_bear = calculate_breadth_persistence(
    breadth_data,
    latest["datetime"],
    lookback_bars=int(breadth_lookback_bars)
)


# Apply latest breadth-adjusted quality.
direction = latest["signal_direction"]
original_quality = latest["raw_signal_quality"]
quality, breadth_quality_note, original_quality = adjust_quality_for_breadth(
    direction,
    original_quality,
    breadth_state
)
regime = latest["regime"]
regime_score = calculate_regime_score(latest, breadth_info)
regime_description = regime_score_label(regime_score)
day_type = classify_day_type(today_df, latest)
gap_type = classify_gap_type(df, today)
rvol_label = interpret_rvol(latest["rvol"])

# Add breadth-adjusted quality column for chart/table review.
df["signal_quality"] = df["raw_signal_quality"]
if direction != "NONE":
    df.loc[df.index[-1], "signal_quality"] = quality

today_df = df[df["date"] == today].copy()

reversal_info = calculate_reversal_risk(
    today_df=today_df,
    latest_row=latest,
    breadth_state=breadth_state,
    breadth_persistence=breadth_persistence
)

reversal_state = reversal_info["reversal_state"]
reversal_direction = reversal_info["reversal_direction"]
reversal_score = reversal_info["reversal_score"]
trend_continuation_status = reversal_info["trend_continuation_status"]
reversal_reasons = reversal_info["reversal_reasons"]

trade_allowed, gate_reasons = evaluate_execution_gate(
    row=latest,
    direction=direction,
    quality=quality,
    min_quality=min_quality,
    regime=regime,
    allow_chop=allow_chop,
    start_trade_time=start_trade_time,
    end_trade_time=end_trade_time,
    breadth_state=breadth_state,
    require_breadth_alignment=require_breadth_alignment,
    reversal_state=reversal_state
)

entry, planned_stop, planned_target = calculate_trade_plan(latest, direction, stop_points, target_points)
suggested_contracts = suggest_contracts(
    trade_allowed,
    quality,
    regime_score,
    latest["rvol"],
    int(max_contracts)
)
final_signal = direction if trade_allowed else "No Trade"

# ============================================================
# Market Context Dashboard
# ============================================================

st.subheader("Market Context Dashboard")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Regime", regime_description)
m2.metric("Regime Score", regime_score)
m3.metric("Signal", final_signal)
m4.metric("Suggested Contracts", suggested_contracts)
m5.metric("Day Type", day_type)

m6, m7, m8, m9 = st.columns(4)
m6.metric("RVOL", f"{latest['rvol']:.2f}" if pd.notna(latest["rvol"]) else "N/A", rvol_label)
m7.metric("Gap Type", gap_type)
m8.metric("Breadth State", breadth_state)
m9.metric("Breadth Persistence", breadth_persistence)

b1, b2, b3, b4 = st.columns(4)
b1.metric("Bullish ETFs", breadth_info["bullish_count"])
b2.metric("Bearish ETFs", breadth_info["bearish_count"])
b3.metric("Net Breadth", breadth_info["net_breadth_score"])
b4.metric("ETFs Available", breadth_info["total_available"])

r1, r2, r3, r4 = st.columns(4)
r1.metric("Reversal Risk", reversal_state)
r2.metric("Reversal Direction", reversal_direction)
r3.metric("Reversal Score", reversal_score)
r4.metric("Trend Continuation", trend_continuation_status)

with st.expander("Reversal Risk Detail", expanded=False):
    for reason in reversal_reasons:
        st.write(f"- {reason}")

with st.expander("14 ETF Breadth Detail", expanded=False):
    if breadth_df.empty:
        st.info("Breadth data unavailable.")
    else:
        st.dataframe(breadth_df, use_container_width=True)

# ============================================================
# Main Status Display
# ============================================================

st.subheader("Current Signal State")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Proxy", symbol)
c2.metric("Last Price", f"{latest['Close']:.2f}")
c3.metric("Raw Regime", regime)
c4.metric("Direction", direction)
c5.metric("Quality", quality, f"Raw: {original_quality}")

c6, c7, c8, c9 = st.columns(4)
c6.metric("Signal Score", int(latest["signal_score"]))
c7.metric("CALL Score", int(latest["call_score"]))
c8.metric("PUT Score", int(latest["put_score"]))
c9.metric("Breadth Adjustment", breadth_quality_note)

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

p1, p2, p3, p4, p5 = st.columns(5)
p1.metric("Entry Reference", f"{entry:.2f}" if pd.notna(entry) else "N/A")
p2.metric("Planned Stop", f"{planned_stop:.2f}" if pd.notna(planned_stop) else "N/A")
p3.metric("Planned Target", f"{planned_target:.2f}" if pd.notna(planned_target) else "N/A")
rr = target_points / stop_points if stop_points > 0 else np.nan
p4.metric("Reward/Risk", f"{rr:.2f}R")
p5.metric("Suggested Contracts", suggested_contracts)

# ============================================================
# Event Logging
# ============================================================

event_record = {
    "timestamp": latest["datetime"],
    "date": latest["date"],
    "symbol": symbol,
    "price": latest["Close"],
    "regime": regime,
    "regime_description": regime_description,
    "regime_score": regime_score,
    "day_type": day_type,
    "gap_type": gap_type,
    "direction": direction,
    "final_signal": final_signal,
    "signal_quality": quality,
    "original_quality": original_quality,
    "signal_score": latest["signal_score"],
    "call_score": latest["call_score"],
    "put_score": latest["put_score"],
    "rvol": latest["rvol"],
    "rvol_label": rvol_label,
    "breadth_state": breadth_state,
    "breadth_bullish_count": breadth_info["bullish_count"],
    "breadth_bearish_count": breadth_info["bearish_count"],
    "net_breadth_score": breadth_info["net_breadth_score"],
    "breadth_persistence": breadth_persistence,
    "reversal_state": reversal_state,
    "reversal_direction": reversal_direction,
    "reversal_score": reversal_score,
    "trend_continuation_status": trend_continuation_status,
    "reversal_reasons": format_reasons(reversal_reasons),
    "suggested_contracts": suggested_contracts,
    "trade_allowed": trade_allowed,
    "gate_reasons": format_reasons(gate_reasons),
    "signal_reasons": latest["signal_reasons"],
    "breadth_quality_note": breadth_quality_note,
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

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=today_df["datetime"],
    open=today_df["Open"],
    high=today_df["High"],
    low=today_df["Low"],
    close=today_df["Close"],
    name="Price"
))

fig.add_trace(go.Scatter(x=today_df["datetime"], y=today_df["vwap"], mode="lines", name="VWAP"))
fig.add_trace(go.Scatter(x=today_df["datetime"], y=today_df["ema_9"], mode="lines", name="EMA 9"))
fig.add_trace(go.Scatter(x=today_df["datetime"], y=today_df["ema_21"], mode="lines", name="EMA 21"))
fig.add_trace(go.Scatter(x=today_df["datetime"], y=today_df["or_high"], mode="lines", name="OR High"))
fig.add_trace(go.Scatter(x=today_df["datetime"], y=today_df["or_low"], mode="lines", name="OR Low"))

# ============================================================
# Signal Marker Overlay
# ============================================================

signal_plot_df = today_df.copy()
signal_plot_df["quality_pass"] = signal_plot_df["signal_quality"].apply(lambda q: quality_passes(q, min_quality))

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

fig.add_trace(go.Scatter(
    x=call_signals["datetime"],
    y=call_signals["Low"] * 0.999,
    mode="markers",
    name="CALL Signal",
    marker=dict(symbol="triangle-up", size=14, color="lime"),
    text=[
        f"CALL<br>Quality: {q}<br>Score: {s}<br>Regime: {r}"
        for q, s, r in zip(call_signals["signal_quality"], call_signals["signal_score"], call_signals["regime"])
    ],
    hoverinfo="text"
))

fig.add_trace(go.Scatter(
    x=put_signals["datetime"],
    y=put_signals["High"] * 1.001,
    mode="markers",
    name="PUT Signal",
    marker=dict(symbol="triangle-down", size=14, color="red"),
    text=[
        f"PUT<br>Quality: {q}<br>Score: {s}<br>Regime: {r}"
        for q, s, r in zip(put_signals["signal_quality"], put_signals["signal_score"], put_signals["regime"])
    ],
    hoverinfo="text"
))

fig.add_trace(go.Scatter(
    x=rejected_signals["datetime"],
    y=rejected_signals["Close"],
    mode="markers",
    name="Rejected Signal",
    marker=dict(symbol="x", size=10, color="gray"),
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


# Reversal risk marker at the latest candle when continuation risk is elevated/confirmed.
if reversal_state in [
    "Bullish reversal risk elevated",
    "Bullish reversal confirmed",
    "Bearish reversal risk elevated",
    "Bearish reversal confirmed"
]:
    fig.add_trace(go.Scatter(
        x=[latest["datetime"]],
        y=[latest["Close"]],
        mode="markers",
        name="Reversal Risk",
        marker=dict(symbol="diamond", size=16, color="yellow"),
        text=[
            f"{reversal_state}<br>Direction: {reversal_direction}<br>Score: {reversal_score}<br>Status: {trend_continuation_status}"
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
    )
)

st.plotly_chart(fig, use_container_width=True)


st.subheader("Full-Day Signal History")

history_df = build_full_day_signal_history(
    df=df,
    today=today,
    breadth_data=breadth_data,
    min_quality=min_quality,
    allow_chop=allow_chop,
    require_breadth_alignment=require_breadth_alignment,
    start_trade_time=start_trade_time,
    end_trade_time=end_trade_time,
    stop_points=stop_points,
    target_points=target_points,
    max_contracts=max_contracts,
    breadth_lookback_bars=breadth_lookback_bars
)

st.dataframe(history_df, use_container_width=True)

csv_history = history_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Full-Day Signal History CSV",
    data=csv_history,
    file_name="0dte_v2_2_full_day_signal_history.csv",
    mime="text/csv"
)

# ============================================================
# Downloads
# ============================================================

st.subheader("Downloads")

if LOG_FILE.exists():
    with open(LOG_FILE, "rb") as f:
        st.download_button("Download Signal Log CSV", data=f, file_name="0dte_live_v2_2_log.csv", mime="text/csv")


# ============================================================
# Footer
# ============================================================

st.divider()
st.caption(
    "V2.2 architecture: regime score, day type, gap type, RVOL interpretation, 14 ETF breadth alignment, "
    "breadth persistence, signal quality grading, reversal-risk monitoring, execution gates, event logging, and candle chart signal overlays."
)
