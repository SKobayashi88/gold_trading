# =========================================================
# AI Market Predictor (TRINITY EDITION: CIO + CRO + PM)
# - CIO: OpenAI (Strategy)
# - CRO: Gemini (Validation)
# - PM: Kelly Criterion & Vol Targeting (Money Management)
# =========================================================

import os
import io
import json
import sqlite3
import hashlib
from datetime import datetime
from typing import List, Literal, Optional, Dict, Any, Tuple

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional Libraries
try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

from scipy.signal import argrelextrema
from pydantic import BaseModel, Field, conint, confloat
from openai import OpenAI

# =========================================================
# 1. Config & Database
# =========================================================
st.set_page_config(page_title="AI Market Predictor (Trinity)", layout="wide", page_icon="🏛️")

if "multi_report" not in st.session_state: st.session_state["multi_report"] = None
if "gemini_analysis" not in st.session_state: st.session_state["gemini_analysis"] = ""
if "calendar_report" not in st.session_state: st.session_state["calendar_report"] = ""
if "validation_result" not in st.session_state: st.session_state["validation_result"] = {}
if "last_logs" not in st.session_state: st.session_state["last_logs"] = {}
if "last_mets" not in st.session_state: st.session_state["last_mets"] = pd.DataFrame()

DB_PATH = "ai_trading_logs.sqlite"

def db_connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, created_at TEXT, ticker TEXT, interval TEXT, lookback_bars INTEGER, model TEXT, payload_json TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, strategy_name TEXT, entry_time TEXT, exit_time TEXT, direction TEXT, entry_price REAL, exit_price REAL, return_pct REAL, bars_held INTEGER, exit_reason TEXT, position_id INTEGER)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS metrics (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, strategy_name TEXT, trades INTEGER, signals INTEGER, win_rate REAL, max_dd REAL, final_equity REAL)""")
    conn.commit()
    return conn

def make_run_id(payload: Dict[str, Any]) -> str:
    s = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:24]

def db_save_run(conn, run_id, ticker, interval, lookback, model, payload):
    conn.execute("INSERT OR REPLACE INTO runs VALUES(?,?,?,?,?,?,?)", (run_id, datetime.now().isoformat(), ticker, interval, lookback, model, json.dumps(payload, default=str, ensure_ascii=False)))
    conn.commit()

def db_save_trades(conn, run_id, strat_name, df):
    if df.empty: return
    rows = [(run_id, strat_name, str(r["entry_time"]), str(r["exit_time"]), r["direction"], float(r["entry_price"]), float(r["exit_price"]), float(r["return_pct"]), int(r["bars_held"]), r["exit_reason"], int(r["position_id"])) for _, r in df.iterrows()]
    conn.executemany("INSERT INTO trades(run_id, strategy_name, entry_time, exit_time, direction, entry_price, exit_price, return_pct, bars_held, exit_reason, position_id) VALUES(?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()

def db_save_metrics(conn, run_id, strat_name, m):
    conn.execute("INSERT INTO metrics(run_id, strategy_name, trades, signals, win_rate, max_dd, final_equity) VALUES(?,?,?,?,?,?,?)", (run_id, strat_name, int(m.get("trades",0)), int(m.get("signals",0)), m.get("win_rate"), m.get("max_dd"), m.get("final_equity")))
    conn.commit()

def db_load_recent_runs(conn, limit=20): return pd.read_sql_query("SELECT run_id, created_at, ticker, interval, lookback_bars, model FROM runs ORDER BY created_at DESC LIMIT ?", conn, params=(limit,))

# =========================================================
# 2. Market Logic
# =========================================================
def _s(x): return x.iloc[:, 0] if isinstance(x, pd.DataFrame) else x
def interval_to_horizon(i): return (48, "H") if i=="1h" else ((26, "W") if i=="1wk" else (30, "D"))
def interval_to_order(i): return 12 if i=="1h" else (4 if i=="1wk" else 8)
def calc_atr(df, n=14):
    tr = pd.concat([df["High"]-df["Low"], (df["High"]-df["Close"].shift(1)).abs(), (df["Low"]-df["Close"].shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()
def confidence_bounds(conf): a=(1-conf)/2; return a*100, (1-a)*100
def simulate_gbm(prices, steps, sims):
    lr = np.log(prices/prices.shift(1)).dropna()
    mu, sigma = lr.mean(), lr.std()
    Z = np.random.normal(0, 1, (sims, steps))
    return float(mu), float(sigma), float(prices.iloc[-1]), float(prices.iloc[-1])*np.exp((mu-0.5*sigma**2 + sigma*Z).sum(axis=1)), lr
def simulate_bootstrap(cur, lr, steps, sims): return cur * np.exp(np.random.choice(lr.values, (sims, steps), replace=True).sum(axis=1))
def detect_patterns(prices, ohlc, interval):
    order = interval_to_order(interval)
    atr = calc_atr(ohlc).dropna()
    tol = float(atr.iloc[-1]*0.6) if not atr.empty else float(prices.iloc[-1]*0.005)
    df = pd.DataFrame({"p": prices.values}, index=prices.index)
    peaks = df.iloc[argrelextrema(df["p"].values, np.greater, order=order)[0]]
    troughs = df.iloc[argrelextrema(df["p"].values, np.less, order=order)[0]]
    sig = []
    if len(peaks)>=2 and abs(float(peaks.iloc[-1].p)-float(peaks.iloc[-2].p))<=tol: sig.append("Double Top")
    if len(troughs)>=2 and abs(float(troughs.iloc[-1].p)-float(troughs.iloc[-2].p))<=tol: sig.append("Double Bottom")
    return peaks, troughs, " / ".join(sig) if sig else "None", tol, order

@st.cache_data(ttl=1800)
def get_news(query: str, max_results: int = 8):
    if not HAS_DDGS: return []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=max_results))
        seen, out = set(), []
        for it in results:
            key = (it.get("url") or "") + "||" + (it.get("title") or "")
            if key in seen: continue
            seen.add(key)
            out.append({"title": it.get("title"), "source": it.get("source"), "short": it.get("body","")[:240]+"..."})
        return out
    except Exception: return []

# =========================================================
# 3. PM Logic (Money Management)
# =========================================================
def calculate_kelly(win_rate, win_loss_ratio):
    if win_loss_ratio <= 0: return 0.0
    # Kelly Formula: f = (bp - q) / b = p - (1-p)/b
    # p = win_rate, b = win_loss_ratio
    kelly = win_rate - (1 - win_rate) / win_loss_ratio
    return max(0.0, kelly)

# =========================================================
# 4. Schemas
# =========================================================
class CostSpec(BaseModel):
    spread_bps: confloat(ge=0, le=200) = 0.0
    slippage_bps: confloat(ge=0, le=200) = 0.0
    fee_bps: confloat(ge=0, le=200) = 0.0

class BacktestSpec(BaseModel):
    direction: Literal["long", "short"]
    entry_price: confloat(gt=0)
    stop_price: confloat(gt=0)
    take_profit_price: confloat(gt=0)
    max_hold_bars: conint(ge=1, le=2000)
    sl_tp_same_bar_policy: Literal["conservative_stop_first", "optimistic_tp_first"] = "conservative_stop_first"
    max_positions: conint(ge=1, le=10) = 1
    cooldown_bars: conint(ge=0, le=500) = 0
    pyramiding_enabled: bool = False
    add_on_profit_pct: confloat(ge=0.0, le=50.0) = 1.0
    max_adds: conint(ge=0, le=10) = 0
    costs: CostSpec = Field(default_factory=CostSpec)

class Strategy(BaseModel):
    name: str
    stance: Literal["Bullish", "Bearish", "Neutral"]
    executive_summary: List[str]
    backtest_spec: BacktestSpec

class MultiStrategyReport(BaseModel):
    ticker: str
    strategies: List[Strategy]

# =========================================================
# 5. Engine
# =========================================================
def bps_to_mult(bps: float) -> float: return bps / 10000.0
def apply_entry_cost(price, direction, costs):
    spread, slip = bps_to_mult(costs.spread_bps) * price * 0.5, bps_to_mult(costs.slippage_bps) * price
    return (price + spread + slip) if direction == "long" else (price - spread - slip)
def apply_exit_cost(price, direction, costs):
    spread, slip = bps_to_mult(costs.spread_bps) * price * 0.5, bps_to_mult(costs.slippage_bps) * price
    return (price - spread - slip) if direction == "long" else (price + spread + slip)

def run_pseudo_backtest_multi(df: pd.DataFrame, spec: BacktestSpec, lookback_bars: int = 400):
    df = df.dropna().copy()
    if len(df) < 80: return pd.DataFrame(), {}
    d = df.tail(int(lookback_bars)).copy()
    idx = d.index
    direction, entry, sl, tp = spec.direction, float(spec.entry_price), float(spec.stop_price), float(spec.take_profit_price)
    max_hold, policy = int(spec.max_hold_bars), spec.sl_tp_same_bar_policy
    max_pos, cooldown = int(spec.max_positions), int(spec.cooldown_bars)
    pyr, add_pct, max_adds = bool(spec.pyramiding_enabled), float(spec.add_on_profit_pct)/100.0, int(spec.max_adds)
    costs, fee_mult = spec.costs, 1.0 - bps_to_mult(spec.costs.fee_bps)

    if direction == "long" and not (sl < entry < tp): return pd.DataFrame(), {}
    if direction == "short" and not (tp < entry < sl): return pd.DataFrame(), {}

    positions, trades = [], []
    next_pos_id, adds_done, last_exit_i = 1, 0, -1e9
    realized_equity, equity_curve, signals = 1.0, [], 0
    
    def portfolio_avg(): return float(np.mean([p["entry_price_eff"] for p in positions])) if positions else 0.0
    def can_enter(i): return (i - last_exit_i) > cooldown
    def open_pos(i, raw, is_add):
        nonlocal next_pos_id, realized_equity
        if len(positions) >= max_pos: return
        eff = apply_entry_cost(raw, direction, costs); realized_equity *= fee_mult
        positions.append({"pid": next_pos_id, "ei": i, "et": idx[i], "entry_price_eff": eff})
        next_pos_id += 1
    def close_all(i, raw, reason):
        nonlocal realized_equity, positions, last_exit_i, adds_done
        if not positions: return
        eff_exit = apply_exit_cost(raw, direction, costs); realized_equity *= fee_mult
        for p in positions:
            ret = (eff_exit - p["entry_price_eff"]) / p["entry_price_eff"] if direction == "long" else (p["entry_price_eff"] - eff_exit) / p["entry_price_eff"]
            realized_equity *= (1.0 + ret)
            trades.append({"position_id": p["pid"], "entry_time": p["et"], "exit_time": idx[i], "direction": direction, "entry_price": p["entry_price_eff"], "exit_price": eff_exit, "return_pct": ret * 100.0, "bars_held": i - p["ei"], "exit_reason": reason})
        positions.clear(); adds_done, last_exit_i = 0, i

    for i in range(len(d)):
        row = d.iloc[i]
        H, L, C = float(row["High"]), float(row["Low"]), float(row["Close"])
        if positions:
            if direction == "long":
                if L <= sl and H >= tp: close_all(i, tp if policy=="optimistic_tp_first" else sl, "TP/SL")
                elif L <= sl: close_all(i, sl, "SL")
                elif H >= tp: close_all(i, tp, "TP")
            else:
                if H >= sl and L <= tp: close_all(i, tp if policy=="optimistic_tp_first" else sl, "TP/SL")
                elif H >= sl: close_all(i, sl, "SL")
                elif L <= tp: close_all(i, tp, "TP")
            if positions and (i - min(p["ei"] for p in positions) >= max_hold): close_all(i, C, "TIME")

        if can_enter(i):
            if len(positions) < max_pos:
                if (direction == "long" and L <= entry) or (direction == "short" and H >= entry): signals += 1; open_pos(i, entry, False)
            if positions and pyr and adds_done < max_adds and len(positions) < max_pos:
                avg = portfolio_avg()
                if direction == "long":
                    trig = avg * (1.0 + add_pct)
                    if H >= trig: open_pos(i, trig, True); adds_done += 1
                else:
                    trig = avg * (1.0 - add_pct)
                    if L <= trig: open_pos(i, trig, True); adds_done += 1

        cur_eq = realized_equity
        if positions:
            temp = realized_equity * fee_mult
            mtm_exit = apply_exit_cost(C, direction, costs)
            for p in positions:
                r = (mtm_exit - p["entry_price_eff"]) / p["entry_price_eff"] if direction == "long" else (p["entry_price_eff"] - mtm_exit) / p["entry_price_eff"]
                temp *= (1.0 + r)
            cur_eq = temp
        equity_curve.append(cur_eq)

    if not trades: return pd.DataFrame(), {"final_equity": equity_curve[-1] if equity_curve else 1.0, "max_dd": 0.0, "trades": 0, "signals": signals}
    tdf = pd.DataFrame(trades)
    eq_arr = np.array(equity_curve)
    dd = (eq_arr / np.maximum.accumulate(eq_arr)) - 1.0
    return tdf, {"trades": len(tdf), "signals": signals, "win_rate": (tdf["return_pct"]>0).mean(), "max_dd": dd.min(), "final_equity": eq_arr[-1], "equity_curve": equity_curve}

def grid_optimize(df, base_spec, lookback, entry_g, sl_g, tp_g, hold_g, cost, max_evals):
    results, evals = [], 0
    total = len(entry_g)*len(sl_g)*len(tp_g)*len(hold_g)
    prog = st.progress(0)
    for i_e, entry in enumerate(entry_g):
        for sl in sl_g:
            for tp in tp_g:
                for hold in hold_g:
                    if evals >= max_evals: break
                    evals += 1
                    prog.progress(min(evals/min(total, max_evals), 1.0))
                    if base_spec.direction=="long" and not(sl<entry<tp): continue
                    if base_spec.direction=="short" and not(tp<entry<sl): continue
                    spec = base_spec.model_copy(deep=True)
                    spec.entry_price, spec.stop_price, spec.take_profit_price, spec.max_hold_bars = entry, sl, tp, int(hold)
                    spec.costs = cost
                    _, m = run_pseudo_backtest_multi(df, spec, lookback)
                    if m.get("trades",0)==0: continue
                    score = m["final_equity"] + (m["max_dd"]*2.0)
                    results.append({"score":score, "entry":entry, "sl":sl, "tp":tp, "hold":hold, "eq":m["final_equity"], "dd":m["max_dd"], "tr":m["trades"]})
    prog.empty()
    return pd.DataFrame(results).sort_values("score", ascending=False)

def walk_forward_optimize(df, base_spec, train_b, test_b, step_b, cost, max_evals):
    folds = []
    df = df.dropna().copy()
    start, fold_id, total_bars = 0, 1, len(df)
    status = st.status("Running WFO...", expanded=True)
    while True:
        train_end = start + train_b
        test_end = train_end + test_b
        if test_end > total_bars: break
        status.write(f"Fold {fold_id}: Train[{start}:{train_end}] Test[{train_end}:{test_end}]")
        train_df, test_df = df.iloc[start:train_end], df.iloc[train_end:test_end]
        price_curr = float(train_df.iloc[-1]["Close"])
        atr_val = float(calc_atr(train_df).iloc[-1])
        if base_spec.direction == "long":
            e_g, s_g, t_g = [price_curr * x for x in [0.995, 1.0]], [price_curr - atr_val * x for x in [1.0, 2.0]], [price_curr + atr_val * x for x in [2.0, 4.0]]
        else:
            e_g, s_g, t_g = [price_curr * x for x in [1.005, 1.0]], [price_curr + atr_val * x for x in [1.0, 2.0]], [price_curr - atr_val * x for x in [2.0, 4.0]]
        opt = grid_optimize(train_df, base_spec, len(train_df), e_g, s_g, t_g, [base_spec.max_hold_bars], cost, max_evals)
        if not opt.empty:
            best = opt.iloc[0]
            spec_test = base_spec.model_copy(deep=True)
            spec_test.entry_price, spec_test.stop_price, spec_test.take_profit_price = best["entry"], best["sl"], best["tp"]
            spec_test.costs = cost
            _, m = run_pseudo_backtest_multi(test_df, spec_test, len(test_df))
            folds.append({"fold": fold_id, "train": f"{start}-{train_end}", "test_eq": m.get("final_equity"), "test_dd": m.get("max_dd"), "best_params": json.dumps(best.to_dict())})
        else: folds.append({"fold": fold_id, "status": "No trades"})
        start += step_b; fold_id += 1
    status.update(label="WFO Complete", state="complete", expanded=False)
    return pd.DataFrame(folds)

# =========================================================
# 6. UI
# =========================================================
st.sidebar.title("🏛️ Institutional AI")
st.sidebar.caption("Trinity: CIO + CRO + PM")

openai_key = st.sidebar.text_input("OpenAI Key", type="password")
gemini_key = st.sidebar.text_input("Gemini Key", type="password")
openai_model = st.sidebar.selectbox("CIO Model", ["gpt-5.1", "gpt-4.5-preview", "gpt-4o", "gpt-4o-mini"], index=2)
gemini_model = st.sidebar.selectbox("CRO Model", ["gemini-2.5-flash-lite", "gemini-2.5-flash"], index=0)

ticker = st.sidebar.selectbox("Asset", ["GC=F", "SI=F", "CL=F", "^GSPC", "^N225", "BTC-USD", "JPY=X"])
interval = st.sidebar.selectbox("Interval", ["1h", "1d", "1wk"])
period = "1mo" if interval == "1h" else "2y"
lookback_bars = st.sidebar.slider("Bars", 100, 4000, 800, 50)
confidence = st.sidebar.slider("Conf %", 80, 99, 90) / 100
sims = st.sidebar.slider("Sims", 2000, 20000, 10000)

st.sidebar.markdown("---")
st.sidebar.caption("Costs (BPS)")
global_costs = CostSpec(spread_bps=st.sidebar.number_input("Spread",0.0,200.0,2.0), slippage_bps=st.sidebar.number_input("Slip",0.0,200.0,1.0), fee_bps=st.sidebar.number_input("Fee",0.0,200.0,0.5))
conn = db_connect()

st.title(f"📊 {ticker} Market Intelligence")

@st.cache_data(ttl=300)
def get_market_data(ticker, period, interval):
    return yf.download(ticker, period=period, interval=interval, progress=False)

with st.spinner("Initializing..."):
    data = get_market_data(ticker, period, interval)

if data.empty: st.error("No Data"); st.stop()

prices = _s(data["Close"])
current_price = float(prices.iloc[-1])
steps, unit = interval_to_horizon(interval)
mu, sigma, cur, mc, lr = simulate_gbm(prices, steps, sims)
bs = simulate_bootstrap(cur, lr, steps, sims)
lo, hi = confidence_bounds(confidence)
lower, upper = float(np.percentile(bs, lo)), float(np.percentile(bs, hi))
peaks, troughs, pat, tol, order = detect_patterns(prices, data, interval)
news_items = get_news(f"{ticker} market news", max_results=8)
news_txt = "\n".join([f"- {i.get('title','')} ({i.get('source','')})" for i in news_items[:8]]) or "No news"

ctx = {"ticker": ticker, "interval": interval, "price": current_price, "vol": sigma*100, "horizon": f"{steps}{unit}", "support": lower, "resistance": upper, "pattern": pat, "news": news_txt}

tabs = st.tabs(["🔮 Dashboard", "📰 Intelligence", "🤖 Strategy (CIO)", "🧪 Backtest (Lab)", "🛡️ Risk (CRO)", "💰 Money Mgmt (PM)", "🗃 History"])

# 1. Dashboard
with tabs[0]:
    st.markdown("""### 📊 市場概況
**このタブの見方:**
- **Current Price**: 現在の市場価格
- **Volatility**: 年率換算ボラティリティ（価格変動の激しさ。高いほどリスクも高い）
- **Support / Resistance**: モンテカルロシミュレーションによる予測価格帯（サポート=下限、レジスタンス=上限）
- **ヒストグラム**: 将来価格の分布予測。Model（理論値）とHistory（過去データ）の2つを表示。赤線が現在価格。
    """)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"{current_price:,.2f}")
    c2.metric("Volatility", f"{sigma*100*(252**0.5):.2f}%")
    c3.metric("Support", f"{lower:,.2f}")
    c4.metric("Resistance", f"{upper:,.2f}")
    fig, ax = plt.subplots(figsize=(10, 3)); ax.hist(mc, bins=60, alpha=0.35, label="Model"); ax.hist(bs, bins=60, alpha=0.35, label="History"); ax.axvline(current_price, c="red"); ax.legend(); st.pyplot(fig)
    macro_tickers = {"US10Y": "^TNX", "DXY": "DX-Y.NYB"}; mdf = pd.DataFrame()
    for k, v in macro_tickers.items(): 
        try: mdf[k] = _s(yf.download(v, period="6mo", progress=False)['Close'])
        except Exception: pass
    if not mdf.empty: st.line_chart((mdf-mdf.mean())/mdf.std())

# 2. Intelligence
with tabs[1]:
    st.markdown("""### 📰 マーケットインテリジェンス
**このタブの見方:**
- **Generate Calendar**: 米国の経済指標カレンダーをGemini AIが生成します。重要な経済イベント（FOMC、雇用統計等）の日程確認に使用。
- **Gemini Briefing**: 選択した資産についてAIが市場分析レポートを作成します。現在の価格動向、ボラティリティ、ニュースを考慮した総合分析。

⚠️ APIキーが必要です（サイドバーで入力）
    """)
    if st.button("Generate Calendar"):
        if not gemini_key: st.error("No Key")
        else:
            try:
                genai.configure(api_key=gemini_key)
                txt = "\n".join([f"{i['title']}" for i in get_news("US economic calendar", 5)])
                st.session_state["calendar_report"] = genai.GenerativeModel(gemini_model).generate_content(f"Make calendar table from: {txt}").text
            except Exception as e:
                if "ResourceExhausted" in str(type(e).__name__) or "429" in str(e) or "quota" in str(e).lower():
                    st.error("🚫 APIレート制限に達しました。1分ほど待ってから再試行してください。")
                else:
                    st.error(f"エラーが発生しました: {e}")
    st.markdown(st.session_state["calendar_report"])
    if st.button("Gemini Briefing"):
        if not gemini_key: st.error("No Key")
        else:
            try:
                genai.configure(api_key=gemini_key)
                st.session_state["gemini_analysis"] = genai.GenerativeModel(gemini_model).generate_content(f"Briefing for {ticker}. Data: {ctx}").text
            except Exception as e:
                if "ResourceExhausted" in str(type(e).__name__) or "429" in str(e) or "quota" in str(e).lower():
                    st.error("🚫 APIレート制限に達しました。1分ほど待ってから再試行してください。")
                else:
                    st.error(f"エラーが発生しました: {e}")
    st.markdown(st.session_state["gemini_analysis"])

# 3. Strategy
with tabs[2]:
    st.subheader("🤖 CIO (Strategy)")
    st.markdown("""**このタブの見方:**
OpenAI（CIO=最高投資責任者）が市場データを分析し、具体的な売買戦略を生成します。
- **stance**: Bullish(強気)/Bearish(弱気)/Neutral(中立)の市場見通し
- **entry_price**: エントリー価格（この価格で仕掛ける）
- **stop_price**: 損切り価格（損失を限定するための撤退ライン）
- **take_profit_price**: 利確価格（利益確定の目標値）
- **max_hold_bars**: 最大保有期間（時間による強制決済）

💡 生成された戦略はBacktestタブで検証できます。
    """)
    n_s = st.slider("Strategies", 1,3,2)
    if st.button("Generate (JSON)"):
        if not openai_key: st.error("No Key")
        else:
            client = OpenAI(api_key=openai_key)
            with st.spinner("CIO thinking..."):
                try:
                    resp = client.beta.chat.completions.parse(model=openai_model, messages=[{"role":"user","content":f"Generate {n_s} strategies for {ticker} based on {ctx}. Return JSON."}], response_format=MultiStrategyReport)
                    st.session_state["multi_report"] = resp.choices[0].message.parsed
                    st.success("Done")
                except Exception as e: st.error(e)
    if st.session_state["multi_report"]: st.json(st.session_state["multi_report"].model_dump())

# 4. Backtest
with tabs[3]:
    st.subheader("🧪 Validation Lab")
    st.markdown("""**このタブの見方（バックテスト結果）:**
過去データでCIOの戦略をシミュレーションし、実際に利益が出たかを検証します。

| 指標 | 意味 | 良い値の目安 |
|------|------|-------------|
| **trades** | 取引回数 | 多いほど信頼性↑ |
| **win_rate** | 勝率 | 50%以上 |
| **max_dd** | 最大ドローダウン | -10%以内が理想 |
| **final_equity** | 最終資産（1.0=元本） | 1.0以上で利益 |

🔧 **Grid Search**: パラメータ最適化　｜　**WFO**: ウォークフォワード最適化（過学習防止）
    """)
    rep = st.session_state.get("multi_report")
    if rep and st.button("Execute Backtest"):
        rid = make_run_id({"ctx":ctx}); l, m = {}, []
        for s in rep.strategies:
            sp = s.backtest_spec.model_copy(deep=True); sp.costs = global_costs
            df, mt = run_pseudo_backtest_multi(data, sp, lookback_bars)
            l[s.name]=df; m.append({"strategy":s.name, **mt})
            db_save_trades(conn, rid, s.name, df); db_save_metrics(conn, rid, s.name, mt)
        st.session_state["last_logs"]=l; st.session_state["last_mets"]=pd.DataFrame(m)
        st.success("Executed")
    if not st.session_state["last_mets"].empty:
        st.dataframe(st.session_state["last_mets"])
        pk = st.selectbox("View", list(st.session_state["last_logs"].keys()))
        row = st.session_state["last_mets"][st.session_state["last_mets"]["strategy"]==pk].iloc[0]
        if "equity_curve" in row: st.line_chart(row["equity_curve"])
        c1, c2 = st.columns(2)
        with c1: 
            if st.button("Run Grid Search"):
                s_obj = next(s for s in rep.strategies if s.name==pk)
                base = s_obj.backtest_spec.model_copy(deep=True); base.costs = global_costs
                cur = current_price
                if base.direction=="long": e,s,t = [cur*0.99, cur], [cur*0.98], [cur*1.01, cur*1.02]
                else: e,s,t = [cur*1.01, cur], [cur*1.02], [cur*0.99, cur*0.98]
                st.dataframe(grid_optimize(data, base, lookback_bars, e, s, t, [base.max_hold_bars], global_costs, 200))
        with c2:
            if st.button("Run WFO"):
                s_obj = next(s for s in rep.strategies if s.name==pk)
                base = s_obj.backtest_spec.model_copy(deep=True); base.costs = global_costs
                st.dataframe(walk_forward_optimize(data, base, 500, 100, 100, global_costs, 100))

# 5. Risk
with tabs[4]:
    st.subheader("🛡️ CRO (Audit)")
    st.markdown("""**このタブの見方:**
Gemini AI（CRO=最高リスク管理責任者）がバックテスト結果を監査し、リスク評価を行います。

**確認ポイント:**
- 戦略の勝率とリスク・リワード比は適切か？
- ドローダウンは許容範囲内か？
- 現在のボラティリティ環境で実行可能か？

⚠️ バックテスト実行後に使用可能になります。
    """)
    if not st.session_state["last_mets"].empty and gemini_key and st.button("Request Audit"):
        genai.configure(api_key=gemini_key)
        mets_json = st.session_state["last_mets"].to_json(orient="records")
        p = f"Role: CRO. Audit these backtest results: {mets_json}. Market Vol: {ctx['vol']}. Verdict?"
        st.session_state["validation_result"] = genai.GenerativeModel(gemini_model).generate_content(p).text
    st.markdown(st.session_state.get("validation_result", ""))

# 6. PM (Money Mgmt)
with tabs[5]:
    st.subheader("💰 Portfolio Manager (Sizing)")
    st.markdown("""**このタブの見方（ポジションサイジング）:**
資金管理の最重要部分。「いくら賭けるか」を科学的に計算します。

| 指標 | 意味 |
|------|------|
| **Win Rate** | 勝率（過去の取引から算出） |
| **Payoff Ratio** | 平均利益÷平均損失（1以上が理想） |
| **Kelly (Full)** | ケリー基準による最適投資比率 |

**推奨サイジング:**
- **Kelly Approach (Half)**: フルケリーの半分（保守的な運用）
- **Vol Targeting**: 目標ボラティリティに基づく調整

💡 ケリー公式: f = p - (1-p)/b（p=勝率, b=ペイオフ比）
    """)
    
    if not st.session_state["last_mets"].empty:
        # Inputs
        acc_balance = st.number_input("Account Balance ($)", 1000, 1000000, 10000)
        risk_per_trade_pct = st.slider("Max Risk per Trade (%)", 0.5, 5.0, 1.0) / 100.0
        target_vol = st.slider("Target Volatility (%)", 10, 100, 30)
        
        pm_target = st.selectbox("Select Strategy to Size", list(st.session_state["last_logs"].keys()))
        
        # Get Metrics
        row = st.session_state["last_mets"][st.session_state["last_mets"]["strategy"]==pm_target].iloc[0]
        trades_df = st.session_state["last_logs"][pm_target]
        
        if not trades_df.empty:
            # Calculate Kelly
            wins = trades_df[trades_df['return_pct'] > 0]
            losses = trades_df[trades_df['return_pct'] <= 0]
            avg_win = wins['return_pct'].mean() if not wins.empty else 0
            avg_loss = abs(losses['return_pct'].mean()) if not losses.empty else 0
            win_rate = row['win_rate']
            
            payoff = avg_win / avg_loss if avg_loss > 0 else 0
            kelly = calculate_kelly(win_rate, payoff)
            half_kelly = kelly / 2.0
            
            # Volatility Adjustment
            current_vol = ctx['vol']
            vol_scaler = target_vol / current_vol if current_vol > 0 else 1.0
            vol_scaler = min(vol_scaler, 2.0) # Cap at 2x leverage
            
            # Display
            c1, c2, c3 = st.columns(3)
            c1.metric("Win Rate", f"{win_rate*100:.1f}%")
            c2.metric("Payoff Ratio", f"{payoff:.2f}")
            c3.metric("Kelly (Full)", f"{kelly*100:.1f}%")
            
            st.markdown("---")
            st.markdown("#### 🎯 Recommended Sizing")
            
            # 1. Risk Based Sizing
            # Stop distance (approx from backtest spec or ATR)
            # Assuming stop is roughly 1 ATR for calculation demo or use recent trade stop dist
            
            rec_risk_amt = acc_balance * risk_per_trade_pct
            st.write(f"**Risk Budget:** ${rec_risk_amt:.2f} (based on {risk_per_trade_pct*100}% risk)")
            
            col_k, col_v = st.columns(2)
            with col_k:
                st.info(f"**Kelly Approach (Half)**\n\nAllocate: **{half_kelly*100:.1f}%** of equity\n\nAmount: **${acc_balance * half_kelly:.2f}**")
            with col_v:
                st.success(f"**Vol Targeting**\n\nScaler: **{vol_scaler:.2f}x** (Target {target_vol}% / Current {current_vol:.1f}%)\n\nAdjusted Risk: **${rec_risk_amt * vol_scaler:.2f}**")
        else:
            st.warning("No trades generated in backtest to analyze.")
    else:
        st.info("Run Backtest first to get metrics.")

# 7. History
with tabs[6]:
    st.subheader("🗃 DB History")
    st.markdown("""**このタブの見方:**
過去に実行したバックテストの履歴を表示します。同じ条件での再検証や、異なるパラメータでの比較分析に活用できます。

| カラム | 意味 |
|--------|------|
| **run_id** | 実行ID（一意の識別子） |
| **created_at** | 実行日時 |
| **ticker** | 対象資産 |
| **interval** | 時間足（1h/1d/1wk） |
| **lookback_bars** | 分析に使用したバー数 |
    """)
    st.dataframe(db_load_recent_runs(conn))
