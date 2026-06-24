"""
做T实时信号面板 — 多股票版
支持: 600760 中航沈飞 / 002281 光迅科技
运行: streamlit run live_signal.py
"""

import warnings
import json
import os
import socket
from datetime import datetime, date, time as dtime, timezone, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

warnings.filterwarnings("ignore")
socket.setdefaulttimeout(10)   # 所有网络请求最多等 10 秒，连不上就快速跳过，避免无限转圈

COMMISSION  = 0.00025 * 2 + 0.0005 + 0.0002 * 2
REFRESH_SEC = 20

# ── 配色（量化终端深色主题 · A股红涨绿跌）──
C_UP     = "#f6465d"   # 涨 / 卖出
C_DOWN   = "#2ebd85"   # 跌 / 买回
C_BLUE   = "#4d8af0"   # VWAP / 品牌色
C_GOLD   = "#f0b90b"   # 警示
C_BG     = "#0a0e17"
C_PANEL  = "#151b2b"
C_BORDER = "#232c42"
C_TEXT   = "#e8ecf4"
C_DIM    = "#8b94a8"
MONO     = "ui-monospace,'SF Mono','JetBrains Mono',monospace"

# ── 时区：A股按北京时间运作，不随用户所在地变化 ──
# 用户可能在墨尔本(UTC+10)等地，本地时钟≠交易时段，必须统一用北京时间(UTC+8)
try:
    from zoneinfo import ZoneInfo
    _BJ_TZ = ZoneInfo("Asia/Shanghai")
except Exception:
    _BJ_TZ = timezone(timedelta(hours=8))   # 中国无夏令时，固定 UTC+8 兜底

# 用户常驻墨尔本：用户本地时间固定按墨尔本算，避免云端服务器(UTC)显示错乱
try:
    _LOCAL_TZ = ZoneInfo("Australia/Melbourne")
except Exception:
    _LOCAL_TZ = timezone(timedelta(hours=10))

def beijing_now() -> datetime:
    """当前北京时间（naive，便于与交易时段比较）"""
    return datetime.now(tz=_BJ_TZ).replace(tzinfo=None)

def local_now() -> datetime:
    """用户本地时间（墨尔本）——不随服务器时区变化"""
    return datetime.now(tz=_LOCAL_TZ).replace(tzinfo=None)


# ═══════════════════════════════════════════════════════
# 交易记录持久化（存文件，重启/跨天不丢；按标的隔离）
#   结构: { sym: {"open": {qty,price,time,date} | None, "history": [ {...} ]} }
# ═══════════════════════════════════════════════════════
_TRADE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "t_trades.json")
# 是否跑在 Streamlit Cloud（其工作目录在 /mount/...）——云端硬盘临时，记录过夜会清
IS_CLOUD = os.path.abspath(__file__).startswith("/mount")

def _load_trades_all() -> dict:
    try:
        with open(_TRADE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_trades_all(d: dict):
    try:
        with open(_TRADE_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def get_sym_rec(sym: str) -> dict:
    rec = _load_trades_all().get(sym)
    if not rec:
        return {"open": None, "history": []}
    rec.setdefault("open", None)
    rec.setdefault("history", [])
    return rec

def put_sym_rec(sym: str, rec: dict):
    d = _load_trades_all()
    d[sym] = rec
    _save_trades_all(d)

def record_sell(sym, qty, px, t, dstr, manual=False):
    rec = get_sym_rec(sym)
    rec["open"] = {"qty": qty, "price": px, "time": t, "date": dstr}
    rec["history"].append({"date": dstr, "time": t, "action": "sell",
                           "qty": qty, "price": px, "net": None, "manual": manual})
    put_sym_rec(sym, rec)

def record_buy(sym, qty, px, t, dstr, net, manual=False):
    rec = get_sym_rec(sym)
    rec["open"] = None
    rec["history"].append({"date": dstr, "time": t, "action": "buy",
                           "qty": qty, "price": px, "net": net, "manual": manual})
    put_sym_rec(sym, rec)

def undo_last_sell(sym):
    rec = get_sym_rec(sym)
    if rec["open"] and rec["history"] and rec["history"][-1]["action"] == "sell":
        rec["history"].pop()
    rec["open"] = None
    put_sym_rec(sym, rec)

# ═══════════════════════════════════════════════════════
# 标的独立配置（2 股票 + 2 ETF）
#   is_etf : 走 ETF 行情接口；dp 报价小数位（股票2 / ETF3）；unit 单位
# ═══════════════════════════════════════════════════════
STOCK_CONFIGS = {
    "600760": {
        "symbol":      "600760",
        "symbol_sina": "sh600760",
        "name":        "中航沈飞",
        "is_etf":      False, "dp": 2, "unit": "股",
        # 占位值；真实持仓走 st.secrets（云端加密，不进仓库）
        "base_qty":    1000,
        "avg_cost":    41.0,
        "t_qty":       300,
        # 策略参数（均值回归型，Hurst=0.456）
        "hurst": 0.456, "amp": "2.80%", "show_floating": False,
        "atr_sell_mult":  0.55,
        "atr_buy_mult":   0.40,
        "stop_loss_pct":  0.008,
        "max_t_per_day":  2,
        # 跳空行为（实测：高开81%续涨，低开73%续跌）
        "gap_high_thr":   0.005,   # 超此值视为强高开
        "gap_low_thr":   -0.005,
        "high_open_mult": 1.25,    # 强高开时提高卖出门槛
        "low_open_mult":  1.25,    # 强低开时提高买回门槛
        "low_open_mode":  "wait",  # 低开等待，不急买回
    },
    "002281": {
        "symbol":      "002281",
        "symbol_sina": "sz002281",
        "name":        "光迅科技",
        "is_etf":      False, "dp": 2, "unit": "股",
        "base_qty":    200,
        "avg_cost":    270.0,
        "t_qty":       100,
        # 策略参数（趋势偏向，Hurst=0.547，振幅6.28%）
        "hurst": 0.547, "amp": "6.28%", "show_floating": True,
        "atr_sell_mult":  0.60,    # 比600760更宽（振幅大）
        "atr_buy_mult":   0.45,
        "stop_loss_pct":  0.012,   # 更宽止损（ATR大）
        "max_t_per_day":  2,
        # 跳空行为（实测：高开70%续涨，低开仅42%续跌→低开反而容易反弹！）
        "gap_high_thr":   0.010,   # 超1%视为强高开（价格高，更难触发）
        "gap_low_thr":   -0.010,
        "high_open_mult": 1.20,
        "low_open_mult":  0.85,    # 低开时反而降低卖出门槛（等反弹卖出）
        "low_open_mode":  "sell_bounce",  # 低开后等反弹卖，不是等买回
    },
    "159583": {
        "symbol":      "159583",
        "symbol_sina": "sz159583",
        "name":        "通信ETF",
        "is_etf":      True, "dp": 3, "unit": "份",
        "base_qty":    10000,
        "avg_cost":    2.100,
        "t_qty":       5000,
        # 策略参数（趋势偏向，Hurst=0.537，振幅4.01%，ATR≈0.103）
        "hurst": 0.537, "amp": "4.01%", "show_floating": True,
        "atr_sell_mult":  0.55,
        "atr_buy_mult":   0.42,
        "stop_loss_pct":  0.010,
        "max_t_per_day":  2,
        # ETF 跳空较小，阈值收紧
        "gap_high_thr":   0.008,
        "gap_low_thr":   -0.008,
        "high_open_mult": 1.20,
        "low_open_mult":  1.20,
        "low_open_mode":  "wait",
    },
    "563230": {
        "symbol":      "563230",
        "symbol_sina": "sh563230",
        "name":        "卫星ETF",
        "is_etf":      True, "dp": 3, "unit": "份",
        "base_qty":    10000,
        "avg_cost":    1.350,
        "t_qty":       10000,
        # 策略参数（趋势偏向，Hurst=0.594，振幅3.56%，ATR≈0.049）
        "hurst": 0.594, "amp": "3.56%", "show_floating": True,
        "atr_sell_mult":  0.58,
        "atr_buy_mult":   0.45,
        "stop_loss_pct":  0.009,
        "max_t_per_day":  2,
        "gap_high_thr":   0.008,
        "gap_low_thr":   -0.008,
        "high_open_mult": 1.20,
        "low_open_mult":  1.20,
        "low_open_mode":  "wait",
    },
}

FORCE_CLOSE = (14, 48)


# ═══════════════════════════════════════════════════════
# 页面设置
# ═══════════════════════════════════════════════════════
st.set_page_config(
    page_title="做T信号面板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st_autorefresh(interval=REFRESH_SEC * 1000, key="live_refresh")

# ═══════════════════════════════════════════════════════
# 全局视觉主题（量化终端深色风）
# ═══════════════════════════════════════════════════════
st.markdown(f"""
<style>
.stApp {{
  background: radial-gradient(1100px 560px at 50% -8%, #16203a 0%, {C_BG} 58%);
}}
.block-container {{ padding-top: 2rem; padding-bottom: 2.5rem; max-width: 1480px; }}
h1,h2,h3 {{ letter-spacing:.3px; }}

/* 指标卡 */
[data-testid="stMetric"] {{
  background: linear-gradient(180deg,#161d2e 0%,#121828 100%);
  border:1px solid {C_BORDER}; border-radius:14px;
  padding:14px 16px 12px; box-shadow:0 2px 12px rgba(0,0,0,.28);
}}
[data-testid="stMetricLabel"] p {{ color:{C_DIM}; font-size:12px; font-weight:600; letter-spacing:.4px; }}
[data-testid="stMetricValue"] {{
  font-family:{MONO}; font-weight:700; font-variant-numeric:tabular-nums; letter-spacing:.5px;
}}
[data-testid="stMetricDelta"] {{ font-size:12.5px; font-weight:600; }}

hr {{ border-color:#1c2335 !important; margin:.7rem 0 !important; }}

/* 按钮 */
.stButton > button {{
  border-radius:12px; font-weight:700; letter-spacing:.3px;
  border:1px solid #2a3450; transition:all .15s ease;
}}
.stButton > button:hover {{ transform:translateY(-1px); box-shadow:0 6px 18px rgba(77,138,240,.25); }}
.stButton > button[kind="primary"] {{ border:none; }}

/* 侧边栏 */
[data-testid="stSidebar"] {{
  background:linear-gradient(180deg,#0d1322 0%,{C_BG} 100%);
  border-right:1px solid #1b2335;
}}

/* expander / alert */
[data-testid="stExpander"] {{ border:1px solid {C_BORDER}; border-radius:12px; background:#121828; }}
[data-testid="stAlert"] {{ border-radius:12px; }}
[data-testid="stCaptionContainer"] p {{ color:{C_DIM}; }}

/* 去掉顶部彩条 & Deploy 按钮，更像成品 */
[data-testid="stDecoration"] {{ display:none; }}
[data-testid="stAppDeployButton"] {{ display:none; }}

/* 顶部股票切换条 */
[data-testid="stSegmentedControl"] {{ margin-bottom:4px; }}
[data-testid="stSegmentedControl"] button {{ font-weight:700; letter-spacing:.3px; }}

@keyframes sigpulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.72}} }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# 股票切换（放页面顶部，手机/收起侧栏也能切）
# ═══════════════════════════════════════════════════════
_keys = list(STOCK_CONFIGS.keys())
selected = st.segmented_control(
    "选择标的",
    options=_keys,
    format_func=lambda k: f"{STOCK_CONFIGS[k]['name']} {k}",
    default=_keys[0],
    key="stock_sel",
    label_visibility="collapsed",
)
if not selected:
    selected = _keys[0]
cfg = dict(STOCK_CONFIGS[selected])

# 真实持仓从 st.secrets 读取（本地 .streamlit/secrets.toml 或云端加密 secrets）
# 读不到就用上面的占位值——保证代码仓库里没有任何真实成本/股数
try:
    _h = st.secrets["holdings"][selected]
    cfg["base_qty"] = int(_h["base_qty"])
    cfg["avg_cost"] = float(_h["avg_cost"])
    cfg["t_qty"]    = int(_h["t_qty"])
except Exception:
    pass


# ═══════════════════════════════════════════════════════
# 侧边栏：配置预览
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📊 做T信号面板")
    st.caption(f"当前：**{cfg['name']}** {selected}")
    st.divider()
    _u = cfg["unit"]
    st.caption("**持仓配置**" + ("　·　ETF" if cfg["is_etf"] else "　·　股票"))
    st.caption(f"底仓: {cfg['base_qty']}{_u}")
    st.caption(f"成本: {cfg['avg_cost']:.{cfg['dp']}f}元")
    st.caption(f"做T量: {cfg['t_qty']}{_u}/次")
    st.caption(f"ATR卖出倍数: {cfg['atr_sell_mult']}")
    st.caption(f"ATR买回倍数: {cfg['atr_buy_mult']}")
    st.caption(f"止损线: {cfg['stop_loss_pct']*100:.1f}%")

    st.divider()
    hurst_val = cfg["hurst"]
    mode = "均值回归" if hurst_val < 0.5 else "趋势偏向"
    low_mode = "低开→等待" if cfg["low_open_mode"] == "wait" else "低开→等反弹卖出"
    st.caption(f"**标的特征**")
    st.caption(f"Hurst: {hurst_val} ({mode})")
    st.caption(f"日均振幅: {cfg['amp']}")
    st.caption(f"低开策略: {low_mode}")

    st.divider()
    # 交易时段换算成用户本地时间（应对墨尔本等时区）
    _bj = beijing_now()
    _local = local_now()
    _offset_h = round((_local - _bj).total_seconds() / 3600)
    def _bj2local(hh, mm):
        return (datetime(2000,1,1,hh,mm) + timedelta(hours=_offset_h)).strftime("%H:%M")
    st.caption("**交易时段（北京 → 本地）**")
    if _offset_h != 0:
        st.caption(f"本地比北京 {'快' if _offset_h>0 else '慢'} {abs(_offset_h)} 小时")
        st.caption(f"上午 09:30–11:30 → {_bj2local(9,30)}–{_bj2local(11,30)}")
        st.caption(f"下午 13:00–15:00 → {_bj2local(13,0)}–{_bj2local(15,0)}")
    else:
        st.caption("上午 09:30–11:30 / 下午 13:00–15:00")
    st.divider()
    st.caption(f"每{REFRESH_SEC}秒自动刷新")


# ═══════════════════════════════════════════════════════
# 会话状态（按股票隔离）
# ═══════════════════════════════════════════════════════
def _state_key(k):
    return f"{selected}_{k}"

def ss(k):
    return st.session_state.get(_state_key(k))

def ss_set(k, v):
    st.session_state[_state_key(k)] = v

# 从文件载入当前标的的交易状态（文件=唯一真相，重启/跨天/换设备都不丢）
today = beijing_now().date().isoformat()
_rec   = get_sym_rec(selected)
_open  = _rec["open"]
_hist  = _rec["history"]

ss_set("t_status",   "waiting_buy" if _open else "waiting_sell")
ss_set("sold_price", float(_open["price"]) if _open else 0.0)
ss_set("sold_time",  _open["time"] if _open else None)
ss_set("sold_date",  _open["date"] if _open else None)
ss_set("sold_qty",   int(_open["qty"]) if _open else cfg["t_qty"])
ss_set("history",    _hist)

# 今日已完成的买回 → 今日做T次数 / 今日盈亏（按北京日期自动归零，不清历史）
_today_buys = [h for h in _hist if h.get("action") == "buy" and h.get("date") == today]
ss_set("t_count", len(_today_buys))
ss_set("pnl",     float(sum((h.get("net") or 0.0) for h in _today_buys)))


# ═══════════════════════════════════════════════════════
# 数据获取
# ═══════════════════════════════════════════════════════
def _daily_df(symbol_sina: str, is_etf: bool):
    """统一日线源：ETF 用 fund_etf_hist_sina，股票用 stock_zh_a_daily。
       均含 date/open/high/low/close 列。"""
    import akshare as ak
    if is_etf:
        df = ak.fund_etf_hist_sina(symbol=symbol_sina)        # date open high low close volume amount
    else:
        df = ak.stock_zh_a_daily(symbol=symbol_sina, adjust="qfq")
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    return df


@st.cache_data(ttl=3600)
def fetch_prev_close(symbol_sina: str, is_etf: bool = False):
    """上一交易日收盘价（用于跳空/涨跌幅计算的兜底）"""
    try:
        df = _daily_df(symbol_sina, is_etf)
        return float(df.iloc[-1]["close"]), str(df.iloc[-1]["date"])[:10]
    except Exception:
        return 0.0, ""


@st.cache_data(ttl=REFRESH_SEC)
def fetch_minute_bars(symbol: str, symbol_sina: str):
    """今日1分钟K线。新浪优先（稳定快），东财为备用。返回 OHLCV+amount，index=时间"""
    import akshare as ak
    today_str = beijing_now().strftime("%Y-%m-%d")

    # ① 新浪（主路径，稳定）
    try:
        df = ak.stock_zh_a_minute(symbol=symbol_sina, period="1", adjust="")
        if df is not None and not df.empty:
            df = df.rename(columns={"day": "time"})
            df["time"] = pd.to_datetime(df["time"])
            df = df[df["time"].dt.strftime("%Y-%m-%d") == today_str]   # 只保留今天
            if not df.empty:
                df.set_index("time", inplace=True)
                for c in ["open", "high", "low", "close", "volume"]:
                    df[c] = df[c].astype(float)
                df["amount"] = df["amount"].astype(float) if "amount" in df \
                    else df["close"] * df["volume"]
                out = df[["open","high","low","close","volume","amount"]]
                out.attrs["src"] = "新浪"
                return out
    except Exception:
        pass

    # ② 东方财富（备用；连不上会在 10 秒内超时跳过）
    try:
        df = ak.stock_zh_a_hist_min_em(
            symbol=symbol, period="1",
            start_date=f"{today_str} 09:30:00",
            end_date=f"{today_str} 15:00:00",
            adjust="",
        )
        if df is not None and not df.empty:
            df.columns = ["time","open","close","high","low","volume","amount",
                          "振幅","涨跌幅","涨跌额","换手率"]
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            out = df[["open","high","low","close","volume","amount"]].astype(float)
            out.attrs["src"] = "东财"
            return out
    except Exception:
        pass

    return pd.DataFrame()


@st.cache_data(ttl=REFRESH_SEC)
def fetch_realtime(symbol: str, symbol_sina: str, is_etf: bool = False):
    import akshare as ak

    # ① 今日分钟线推导盘中价（新浪，稳定快速——主路径）
    try:
        bars = fetch_minute_bars(symbol, symbol_sina)
        if not bars.empty:
            pre_close, _ = fetch_prev_close(symbol_sina, is_etf)
            last = bars.iloc[-1]
            price = float(last["close"])
            chg = (price - pre_close) / pre_close * 100 if pre_close > 0 else 0.0
            src = bars.attrs.get("src", "分钟线")
            last_t = bars.index[-1].strftime("%H:%M")
            return {
                "price":      price,
                "open":       float(bars.iloc[0]["open"]),
                "high":       float(bars["high"].max()),
                "low":        float(bars["low"].min()),
                "pre_close":  pre_close,
                "volume":     float(bars["volume"].sum()),
                "amount":     float(bars["amount"].sum()),
                "change_pct": chg,
                "source":     f"盘中推导·{src}(截至{last_t})",
            }
    except Exception:
        pass

    # ② 东方财富实时快照（仅股票，补充；连不上会在 10 秒内超时跳过）
    if not is_etf:
        try:
            df = ak.stock_zh_a_spot_em()
            row = df[df["代码"] == symbol]
            if not row.empty:
                r = row.iloc[0]
                price = float(r["最新价"])
                if price > 0:
                    return {
                        "price":      price,
                        "open":       float(r["今开"]),
                        "high":       float(r["最高"]),
                        "low":        float(r["最低"]),
                        "pre_close":  float(r["昨收"]),
                        "volume":     float(r["成交量"]) * 100,
                        "amount":     float(r["成交额"]),
                        "change_pct": float(r["涨跌幅"]),
                        "source":     "实时(东财)",
                    }
        except Exception:
            pass

    # ③ 最后兜底：上一交易日收盘
    try:
        df2 = _daily_df(symbol_sina, is_etf)
        last, prev = df2.iloc[-1], df2.iloc[-2]
        chg = (float(last["close"]) - float(prev["close"])) / float(prev["close"]) * 100
        return {
            "price":      float(last["close"]),
            "open":       float(last["open"]),
            "high":       float(last["high"]),
            "low":        float(last["low"]),
            "pre_close":  float(prev["close"]),
            "volume":     float(last["volume"]),
            "amount":     float(last["amount"]),
            "change_pct": chg,
            "source":     f"上一收盘 ({str(last['date'])[:10]})",
        }
    except Exception as e:
        return {"_error": str(e)}


@st.cache_data(ttl=3600)
def fetch_atr(symbol_sina: str, is_etf: bool = False, period: int = 14):
    fallback = {"sh600760": 1.32, "sz002281": 16.36, "sz159583": 0.103, "sh563230": 0.049}
    try:
        df = _daily_df(symbol_sina, is_etf)
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
        return round(float(tr.rolling(period).mean().iloc[-1]), 4)
    except Exception:
        return fallback.get(symbol_sina, 1.0)


# ═══════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════
def calc_vwap(bars: pd.DataFrame) -> float:
    if bars.empty or bars["volume"].sum() == 0:
        return 0.0
    return float((bars["close"] * bars["volume"]).sum() / bars["volume"].sum())


def is_trading_time(now: datetime) -> tuple[bool, str]:
    if now.weekday() >= 5:
        return False, "周末休市"
    t = now.time()
    if dtime(9, 30) <= t <= dtime(9, 45):
        return False, "开盘消化期 · 不操作"
    if dtime(9, 45) <= t <= dtime(11, 30):
        return True,  "主交易窗口 ✓"
    if dtime(11, 30) < t < dtime(13, 0):
        return False, "午间休市 · 暂停"
    if dtime(13, 0) <= t <= dtime(13, 20):
        return False, "午盘消化期 · 不操作"
    if dtime(13, 20) <= t <= dtime(14, 45):
        return True,  "第二交易窗口 ✓"
    if dtime(14, 45) <= t < dtime(14, 48):
        return False, "⚠️ 即将强制平仓"
    if t < dtime(9, 30) or t > dtime(15, 0):
        return False, "非交易时段"
    return False, "收盘"


def force_close_check(now: datetime) -> bool:
    t = now.time()
    return dtime(FORCE_CLOSE[0], FORCE_CLOSE[1]) <= t <= dtime(15, 0)


def compute_signal(price, vwap, atr, open_price, pre_close, now, cfg):
    """核心信号逻辑，完全参数化，支持不同股票特征"""
    gap_pct = (open_price - pre_close) / pre_close if pre_close > 0 else 0
    tradable, time_reason = is_trading_time(now)
    t_qty  = cfg["t_qty"]
    max_t  = cfg["max_t_per_day"]
    dp     = cfg["dp"]

    # 基础卖出/买回目标
    sell_target = round(vwap + cfg["atr_sell_mult"] * atr, dp)
    buy_target  = round(vwap - cfg["atr_buy_mult"]  * atr, dp)

    # 跳空调整
    if gap_pct > cfg["gap_high_thr"]:
        # 强高开：提高卖出门槛（趋势可能延续）
        sell_target = round(vwap + cfg["atr_sell_mult"] * atr * cfg["high_open_mult"], dp)

    if gap_pct < cfg["gap_low_thr"]:
        if cfg["low_open_mode"] == "sell_bounce":
            # 002281 低开后反弹概率高(58%)→ 降低卖出门槛，等反弹时卖
            sell_target = round(vwap + cfg["atr_sell_mult"] * atr * cfg["low_open_mult"], dp)
        else:
            # 600760 低开后续跌(73%)→ 提高买回门槛，等更低价买
            buy_target = round(vwap - cfg["atr_buy_mult"] * atr * cfg["low_open_mult"], dp)

    # 止损优先级最高
    if ss("t_status") == "waiting_buy" and ss("sold_price") > 0:
        sold = ss("sold_price")
        stop_price = round(sold * (1 + cfg["stop_loss_pct"]), dp)
        if price >= stop_price:
            return "STOP_LOSS", sell_target, buy_target, gap_pct, \
                   f"价格{price:.{dp}f}触及止损线{stop_price:.{dp}f}，立即买回"
        if force_close_check(now):
            return "FORCE_CLOSE", sell_target, buy_target, gap_pct, "14:48强制平仓"

    if not tradable:
        return "NO_TIME", sell_target, buy_target, gap_pct, time_reason

    if ss("t_count") >= max_t:
        return "HOLD", sell_target, buy_target, gap_pct, \
               f"今日已做T {ss('t_count')}次，达到上限"

    # 等待卖出
    if ss("t_status") == "waiting_sell":
        if price >= sell_target:
            dist_pct = (price - sell_target) / sell_target * 100
            hint = ""
            if gap_pct < cfg["gap_low_thr"] and cfg["low_open_mode"] == "sell_bounce":
                hint = " ｜ 低开反弹卖出点"
            return "SELL", sell_target, buy_target, gap_pct, \
                   f"价格{price:.{dp}f}到达卖出目标{sell_target:.{dp}f}（超目标{dist_pct:.2f}%）{hint}"
        else:
            remain = sell_target - price
            return "HOLD", sell_target, buy_target, gap_pct, \
                   f"距卖出目标还差 {remain:.{dp}f}元 ({remain/sell_target*100:.2f}%)"

    # 等待买回
    if ss("t_status") == "waiting_buy":
        if price <= buy_target:
            sold = ss("sold_price")
            profit_est = (sold - price) * t_qty * (1 - COMMISSION)
            return "BUY", sell_target, buy_target, gap_pct, \
                   f"价格{price:.{dp}f}到达买回目标{buy_target:.{dp}f}，预计净利 {profit_est:.0f}元"
        else:
            remain = price - buy_target
            cur_pnl = (ss("sold_price") - price) * t_qty
            return "HOLD", sell_target, buy_target, gap_pct, \
                   f"距买回目标还差 {remain:.{dp}f}元 · T仓浮动 {cur_pnl:+.0f}元"

    return "HOLD", sell_target, buy_target, gap_pct, "等待机会"


# ═══════════════════════════════════════════════════════
# UI 样式定义
# ═══════════════════════════════════════════════════════
SIGNAL_STYLES = {
    # type:        (emoji, label,           desc,                      accent)
    "SELL":        ("📤", "卖出信号",        f"现在卖出 {cfg['t_qty']}{cfg['unit']}", C_UP),
    "BUY":         ("📥", "买回信号",        f"现在买回 {cfg['t_qty']}{cfg['unit']}", C_DOWN),
    "STOP_LOSS":   ("🚨", "止损！立即买回",   "价格反向 · 市价买回",        C_UP),
    "FORCE_CLOSE": ("⏰", "强制平仓",        "14:48 · 市价买回",          C_GOLD),
    "HOLD":        ("⏳", "观望",            "",                          C_DIM),
    "NO_TIME":     ("🕐", "非操作时段",       "",                          C_DIM),
}

def render_signal_box(sig_type, analysis):
    emoji, label, desc, accent = SIGNAL_STYLES[sig_type]
    body   = desc if sig_type not in ("HOLD", "NO_TIME") else analysis
    active = sig_type in ("SELL", "BUY", "STOP_LOSS", "FORCE_CLOSE")
    if active:
        bg    = f"linear-gradient(160deg,{accent}26 0%,#121828 62%)"
        glow  = f"box-shadow:0 0 0 1px {accent}66,0 12px 44px {accent}33;"
        bd    = f"{accent}77"
        lab_c = accent
        pulse = "animation:sigpulse 1.6s ease-in-out infinite;"
    else:
        bg    = "linear-gradient(160deg,#161d2e,#121828)"
        glow  = "box-shadow:0 6px 24px rgba(0,0,0,.30);"
        bd    = C_BORDER
        lab_c = C_TEXT
        pulse = ""
    html = (
        f'<div style="background:{bg};border:1px solid {bd};border-radius:18px;'
        f'padding:26px 28px;text-align:center;margin-bottom:14px;{glow}">'
        f'<div style="font-size:46px;line-height:1;margin-bottom:8px;{pulse}">{emoji}</div>'
        f'<div style="font-size:30px;font-weight:800;color:{lab_c};letter-spacing:1.5px;'
        f'margin-bottom:8px">{label}</div>'
        f'<div style="font-size:14px;color:{C_DIM};line-height:1.55">{body}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def section_title(text, icon=""):
    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:8px;margin:2px 0 12px">
              <span style="width:3px;height:15px;background:{C_BLUE};border-radius:2px"></span>
              <span style="font-size:13px;font-weight:700;color:{C_TEXT};letter-spacing:.6px">{icon}{text}</span>
            </div>""",
        unsafe_allow_html=True,
    )


def render_kline(bars: pd.DataFrame, current_price: float,
                 vwap: float, sell_target: float, buy_target: float, dp: int = 2):
    if bars.empty:
        st.caption("📈 K线数据将在 09:30 开盘后自动加载")
        return

    # 数据新鲜度提示（确认实时数据在流动）
    last_t = bars.index[-1].strftime("%H:%M")
    st.caption(f"📈 实时分钟K线 · 共 {len(bars)} 根 · 截至 {last_t}（北京）")

    # Running VWAP from minute bars (more accurate than spot VWAP)
    cum_v = bars["volume"].cumsum().replace(0, np.nan)
    vwap_series = (bars["close"] * bars["volume"]).cumsum() / cum_v

    fig = go.Figure()

    # Candlestick — A股惯例：红涨绿跌
    fig.add_trace(go.Candlestick(
        x=bars.index,
        open=bars["open"],
        high=bars["high"],
        low=bars["low"],
        close=bars["close"],
        name="K线",
        increasing_line_color=C_UP,
        increasing_fillcolor=C_UP,
        decreasing_line_color=C_DOWN,
        decreasing_fillcolor=C_DOWN,
        showlegend=False,
        whiskerwidth=0,
        line_width=1.1,
    ))

    # VWAP 线
    fig.add_trace(go.Scatter(
        x=bars.index,
        y=vwap_series,
        mode="lines",
        name="VWAP",
        line=dict(color=C_BLUE, width=1.6, dash="dot"),
    ))

    # 卖出目标线
    fig.add_hline(
        y=sell_target,
        line_color=C_UP, line_dash="dash", line_width=1.2,
        annotation_text=f"卖 {sell_target:.{dp}f}",
        annotation_position="right",
        annotation_font=dict(color=C_UP, size=11),
    )

    # 买回目标线
    fig.add_hline(
        y=buy_target,
        line_color=C_DOWN, line_dash="dash", line_width=1.2,
        annotation_text=f"买 {buy_target:.{dp}f}",
        annotation_position="right",
        annotation_font=dict(color=C_DOWN, size=11),
    )

    # 当前价格线
    fig.add_hline(
        y=current_price,
        line_color="rgba(232,236,244,0.45)", line_dash="dot", line_width=0.9,
    )

    fig.update_layout(
        height=300,
        margin=dict(l=0, r=72, t=8, b=20),
        showlegend=True,
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=10, color=C_DIM)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C_DIM),
        hovermode="x unified",
        xaxis=dict(
            rangeslider=dict(visible=False),
            showgrid=True,
            gridcolor="rgba(255,255,255,0.045)",
            tickformat="%H:%M",
            type="date",
            linecolor="rgba(255,255,255,0.08)",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.045)",
            tickformat=f".{dp}f",
            side="right",
            zeroline=False,
        ),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ═══════════════════════════════════════════════════════
# 主界面
# ═══════════════════════════════════════════════════════
now = beijing_now()

with st.spinner("获取行情…"):
    spot = fetch_realtime(cfg["symbol"], cfg["symbol_sina"], cfg["is_etf"])
    bars = fetch_minute_bars(cfg["symbol"], cfg["symbol_sina"])
    atr  = fetch_atr(cfg["symbol_sina"], cfg["is_etf"])

# 数据错误处理
if spot is None or "_error" in (spot or {}):
    err = (spot or {}).get("_error", "未知")
    st.markdown(f"### {cfg['name']}  `{cfg['symbol']}`")
    st.info("📴 当前为非交易时间（周末/盘后），面板将在下一个交易日 09:30 后恢复实时数据。")
    with st.expander("技术详情"):
        st.caption(f"错误: {err}")
    st.stop()

price      = spot["price"]
open_price = spot["open"]
high_price = spot["high"]
low_price  = spot["low"]
pre_close  = spot["pre_close"]
change_pct = spot["change_pct"]
data_source = spot.get("source", "实时")
dp   = cfg["dp"]      # 报价小数位（股票2 / ETF3）
unit = cfg["unit"]    # 单位（股 / 份）

vwap = calc_vwap(bars) if not bars.empty else price

sig_type, sell_target, buy_target, gap_pct, analysis = compute_signal(
    price, vwap, atr, open_price, pre_close, now, cfg
)

# ── 顶部 Hero ───────────────────────────────────────
tradable, time_reason = is_trading_time(now)
local_t = local_now().strftime("%H:%M")
chg_c   = C_UP if change_pct >= 0 else C_DOWN
arrow   = "▲" if change_pct >= 0 else "▼"
if gap_pct > cfg["gap_high_thr"]:
    gap_txt, gap_c = "强高开", C_UP
elif gap_pct < cfg["gap_low_thr"]:
    gap_txt = "低开反弹机会" if cfg["low_open_mode"] == "sell_bounce" else "低开等待"
    gap_c   = C_DOWN
else:
    gap_txt, gap_c = "平开", C_DIM
status_c = C_DOWN if tradable else C_DIM

st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;gap:18px;flex-wrap:wrap;
     background:linear-gradient(120deg,#141b2c 0%,#0e1422 100%);border:1px solid {C_BORDER};
     border-radius:18px;padding:16px 26px;margin-bottom:14px;box-shadow:0 4px 20px rgba(0,0,0,.25)">
  <div style="display:flex;align-items:baseline;gap:12px;flex-wrap:wrap">
    <span style="font-size:25px;font-weight:800;color:{C_TEXT}">{cfg['name']}</span>
    <span style="font-family:{MONO};font-size:14px;color:{C_DIM};background:#0e1422;
          border:1px solid {C_BORDER};padding:2px 10px;border-radius:8px">{cfg['symbol']}</span>
    <span style="font-size:12px;color:{gap_c};border:1px solid {gap_c}55;background:{gap_c}14;
          padding:2px 11px;border-radius:20px">{gap_txt}</span>
  </div>
  <div style="display:flex;align-items:baseline;gap:12px">
    <span style="font-family:{MONO};font-size:38px;font-weight:800;color:{chg_c};letter-spacing:.5px">{price:.{dp}f}</span>
    <span style="font-size:17px;font-weight:700;color:{chg_c}">{arrow} {change_pct:+.2f}%</span>
  </div>
  <div style="text-align:right;min-width:190px">
    <div style="font-size:12px;color:{C_DIM}">北京 <b style="color:{C_TEXT};font-family:{MONO}">{now.strftime('%H:%M:%S')}</b> · 本地 {local_t}</div>
    <div style="font-size:13px;color:{status_c};font-weight:700;margin-top:3px">{'● ' if tradable else '○ '}{time_reason}</div>
    <div style="font-size:11px;color:{C_DIM};margin-top:2px">{data_source} · ATR {atr:.{dp}f}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── 三列主体 ──────────────────────────────────────
left, center, right = st.columns([1.4, 2, 1.4])

with left:
    section_title("关键价位")
    st.metric("VWAP（均价线）", f"{vwap:.{dp}f}")
    delta_sell = sell_target - price
    st.metric("卖出目标", f"{sell_target:.{dp}f}",
              f"还差 {delta_sell:+.{dp}f}元" if delta_sell > 0 else "✅ 已到达",
              delta_color="inverse")
    delta_buy = price - buy_target
    st.metric("买回目标", f"{buy_target:.{dp}f}",
              f"还差 {delta_buy:+.{dp}f}元" if delta_buy > 0 else "✅ 已到达")
    st.divider()
    st.caption(f"今日 最高 `{high_price:.{dp}f}` / 最低 `{low_price:.{dp}f}`")

    # 002281 专属：低开提示
    if cfg["symbol"] == "002281" and gap_pct < cfg["gap_low_thr"]:
        st.info("💡 **低开反弹机会**\n\n"
                "实测低开后58%概率反弹，\n"
                "等价格从低点反弹后卖出，\n"
                "再等回落时买回", icon=None)

with center:
    render_signal_box(sig_type, analysis)

    # 实时K线（分钟K + VWAP + 目标线）
    render_kline(bars, price, vwap, sell_target, buy_target, cfg["dp"])

    # 价格位置进度条（绿→中性→红 渐变轨 + 发光滑块）
    if buy_target < sell_target:
        pct  = max(0, min(100, (price - buy_target) / (sell_target - buy_target) * 100))
        knob = C_UP if pct >= 85 else (C_DOWN if pct <= 15 else C_BLUE)
        st.markdown(
            f"""<div style="font-size:11px;color:{C_DIM};margin-bottom:9px;letter-spacing:.3px">
                  价格在目标区间的位置
                </div>
                <div style="position:relative;height:8px;border-radius:6px;
                     background:linear-gradient(90deg,{C_DOWN}66 0%,{C_BORDER} 50%,{C_UP}66 100%)">
                  <div style="position:absolute;left:calc({pct:.0f}% - 7px);top:-5px;width:14px;height:18px;
                       border-radius:5px;background:{knob};border:2px solid {C_BG};
                       box-shadow:0 0 12px {knob}99"></div>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:11px;
                     color:{C_DIM};margin-top:11px;font-family:{MONO}">
                  <span style="color:{C_DOWN}">买 {buy_target:.{dp}f}</span>
                  <span>VWAP {vwap:.{dp}f}</span>
                  <span style="color:{C_UP}">卖 {sell_target:.{dp}f}</span>
                </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # 操作按钮
    t_qty   = cfg["t_qty"]
    now_str = now.strftime("%H:%M:%S")
    if sig_type == "SELL" and ss("t_status") == "waiting_sell":
        if st.button(f"✅ 我已卖出 {t_qty}{unit}（按卖出信号）", use_container_width=True, type="primary"):
            record_sell(selected, t_qty, price, now_str, today)
            st.rerun()

    if sig_type in ("BUY", "STOP_LOSS", "FORCE_CLOSE") and ss("t_status") == "waiting_buy":
        labels = {"BUY": f"✅ 我已买回 {ss('sold_qty')}{unit}（按买回信号）",
                  "STOP_LOSS": f"🚨 我已止损买回 {ss('sold_qty')}{unit}",
                  "FORCE_CLOSE": f"⏰ 已强制平仓买回 {ss('sold_qty')}{unit}"}
        if st.button(labels[sig_type], use_container_width=True, type="primary"):
            sqty = ss("sold_qty")
            net  = (ss("sold_price") - price) * sqty * (1 - COMMISSION)
            record_buy(selected, sqty, price, now_str, today, round(net))
            st.rerun()

    # 撤销卖出：挂单未成交 / 记错了 → 退回等待卖出状态（删掉那条卖出，不计盈亏）
    if ss("t_status") == "waiting_buy":
        if st.button("↩️ 撤销这笔卖出（挂单未成交/记错）", use_container_width=True):
            undo_last_sell(selected)
            st.rerun()

    if IS_CLOUD:
        st.caption("📱 手机/云端仅供看盘 —— 买卖记录请在**电脑端**操作（此处记录过夜会被清空）")

    with st.expander("手动填入实际成交价格"):
        actual = st.number_input("实际成交价", value=float(price), step=10.0 ** (-dp), format=f"%.{dp}f")
        if ss("t_status") == "waiting_sell":
            if st.button("记录卖出"):
                record_sell(selected, t_qty, actual, now_str, today, manual=True)
                st.rerun()
        else:
            if st.button("记录买回"):
                sqty = ss("sold_qty")
                net  = (ss("sold_price") - actual) * sqty * (1 - COMMISSION)
                record_buy(selected, sqty, actual, now_str, today, round(net), manual=True)
                st.rerun()

with right:
    section_title("今日状态")
    remain = cfg["max_t_per_day"] - ss("t_count")
    st.metric("剩余可做T次数", f"{remain} / {cfg['max_t_per_day']}")
    pnl_today = ss("pnl")
    cost_reduced = pnl_today / cfg["base_qty"]
    st.metric("今日累计净盈亏", f"{pnl_today:+.0f} 元",
              f"降成本 {cost_reduced:.4f}元/{unit}")
    st.divider()

    if ss("t_status") == "waiting_buy" and ss("sold_price") > 0:
        sold = ss("sold_price"); sqty = ss("sold_qty")
        cur_p = (sold - price) * sqty
        stop = round(sold * (1 + cfg["stop_loss_pct"]), dp)
        carry = "" if ss("sold_date") == today else f"  ·  {ss('sold_date')}卖出·隔日持有"
        st.warning(f"**T仓挂起中**{carry}\n\n"
                   f"卖出 {sqty}{unit} @ {sold:.{dp}f}  ·  {ss('sold_time')}\n\n"
                   f"当前浮动 {cur_p:+.0f}元\n\n"
                   f"止损线 {stop:.{dp}f}")
    else:
        st.info("当前无开仓T仓\n\n等待卖出机会")

    st.divider()
    new_cost = cfg["avg_cost"] - cost_reduced
    st.metric("当前持仓成本", f"{new_cost:.{dp}f} 元",
              f"已降 {cost_reduced:.4f}元/{unit}" if cost_reduced > 0 else "今日尚未降本")

    # 持仓浮动盈亏（盈利/亏损都显示）
    if cfg.get("show_floating"):
        floating_pnl = (price - cfg["avg_cost"]) * cfg["base_qty"]
        floating_pct = (price / cfg["avg_cost"] - 1) * 100
        st.metric("持仓浮动盈亏", f"{floating_pnl:+.0f} 元", f"{floating_pct:+.1f}%")

    st.divider()
    section_title("交易记录")
    hist = ss("history") or []
    if hist:
        for h in reversed(hist[-8:]):
            act    = "📤 卖出" if h.get("action") == "sell" else "📥 买回"
            net    = h.get("net")
            nettxt = f" · 净 {net:+.0f}元" if net is not None else ""
            mtag   = " ·手动" if h.get("manual") else ""
            st.caption(f"{h.get('date','')} {h.get('time','')} {act} "
                       f"{h.get('qty')}{unit} @ {float(h.get('price',0)):.{dp}f}{nettxt}{mtag}")
        st.caption(f"——— 共 {len(hist)} 笔（显示最近 8 笔）")
    else:
        st.caption("暂无交易记录")

# ── 底部 ──────────────────────────────────────────
st.divider()
low_mode_hint = ("⚡ 002281低开特殊逻辑：降低卖出门槛等反弹 | " if cfg["symbol"] == "002281" and gap_pct < cfg["gap_low_thr"] else "")
st.caption(f"📊 {analysis}  |  {low_mode_hint}卖倍数={cfg['atr_sell_mult']} · 买倍数={cfg['atr_buy_mult']} · ATR14={atr:.{dp}f}")
