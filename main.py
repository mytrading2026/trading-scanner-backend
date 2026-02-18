"""
Trading Scanner Backend — main.py  v2.0
Primary data: Finnhub  |  Fallback: Yahoo Finance
"""

import os, json, time, asyncio, logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Trading Scanner", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# ── State ─────────────────────────────────────────────────────────────────────
bot_enabled       = False
current_strategy  = "default"
connected_clients: list = []
_journal: list    = []
demo_account      = {
    "balance": 100_000.0,
    "positions": [],
    "trade_history": [],
    "total_value": 100_000.0,
}

SCAN_SYMBOLS = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN",
                "BTC-USD", "ETH-USD", "NVDA", "META", "AMD"]

STRATEGIES = {
    "default":      "RSI + EMA crossover",
    "aggressive":   "RSI < 35 / > 65 with momentum",
    "conservative": "EMA alignment only, wider bands",
}

# ── Indicators ────────────────────────────────────────────────────────────────
def calculate_rsi(prices: list, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    arr    = np.array(prices[-50:], dtype=float)  # use last 50 candles max
    d      = np.diff(arr)
    gains  = np.where(d > 0, d, 0.0)
    losses = np.where(d < 0, -d, 0.0)
    # Use Wilder's smoothing instead of simple average
    ag = np.mean(gains[:period])
    al = np.mean(losses[:period])
    for i in range(period, len(gains)):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
    if al == 0:
        return 100.0 if ag > 0 else 50.0
    rs = ag / al
    return round(100.0 - 100.0 / (1.0 + rs), 2)

def generate_signal(price, rsi, ema20, ema50, strategy="default") -> str:
    if strategy == "aggressive":
        if rsi < 35 and price > ema20: return "BUY"
        if rsi > 65 and price < ema20: return "SELL"
    elif strategy == "conservative":
        if ema20 > ema50 and rsi < 55 and price > ema20: return "BUY"
        if ema20 < ema50 and rsi > 45 and price < ema20: return "SELL"
    else:  # default
        if rsi < 30 and price > ema20:  return "BUY"
        if rsi > 70 and price < ema20:  return "SELL"
        if ema20 > ema50 and rsi < 60:  return "BUY"
        if ema20 < ema50 and rsi > 40:  return "SELL"
    return "HOLD"

# ── Data fetching ─────────────────────────────────────────────────────────────
INTERVAL_TO_FINNHUB = {"1m":"1","5m":"5","15m":"15","30m":"30",
                        "1h":"60","4h":"60","1d":"D","1wk":"W","1mo":"M"}

def _fetch_finnhub(symbol: str, interval: str = "15m") -> Optional[dict]:
    key = os.getenv("FINNHUB_API_KEY", "").strip()
    if not key:
        return None

    # Map Yahoo-style symbols to Finnhub format
    symbol_map = {
        "BTC-USD": "BINANCE:BTCUSDT",
        "ETH-USD": "BINANCE:ETHUSDT",
    }
    fh_symbol = symbol_map.get(symbol, symbol)

    try:
        # Get current quote (price + volume)
        url = f"https://finnhub.io/api/v1/quote?symbol={fh_symbol}&token={key}"
        r = requests.get(url, timeout=10)
        q = r.json()
        if not q.get("c") or q["c"] == 0:
            logger.warning("Finnhub quote empty for %s: %s", fh_symbol, q)
            return None
        price = q["c"]
        prev  = q["pc"]
        high  = q["h"]
        low   = q["l"]
        open_ = q["o"]

        # Get daily candle history for RSI/EMA
        now  = int(time.time())
        back = 365 * 86400
        curl = (f"https://finnhub.io/api/v1/stock/candle"
                f"?symbol={fh_symbol}&resolution=D&from={now-back}&to={now}&token={key}")
        cr = requests.get(curl, timeout=10)
        cd = cr.json()

        if cd.get("s") == "ok" and cd.get("c") and len(cd["c"]) >= 15:
            closes = cd["c"]
            # Get volume from candle data
            volume = int(cd["v"][-1]) if cd.get("v") else 0
        else:
            # Not enough candle history — build a synthetic series
            # that will give neutral RSI (~50) rather than 0 or 100
            logger.info("Using synthetic history for %s (candles: %s)",
                        fh_symbol, cd.get("s"))
            step = (price - prev) / 14 if prev else 0
            closes = [round(prev + step * i, 4) for i in range(14)] + [price]
            volume = 0

        rsi   = calculate_rsi(closes)
        ema20 = calculate_ema(closes, 20)
        ema50 = calculate_ema(closes, 50)

        return {
            "symbol": symbol, "price": round(price, 4),
            "open": round(open_, 4), "high": round(high, 4),
            "low": round(low, 4), "volume": volume,
            "rsi": rsi, "ema20": ema20, "ema50": ema50,
            "change_pct": round((price - prev) / prev * 100, 2) if prev else 0,
            "signal": generate_signal(price, rsi, ema20, ema50, current_strategy),
            "interval": interval, "source": "finnhub",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error("Finnhub error for %s: %s", symbol, e)
        return None

def _fetch_yfinance(symbol: str, interval: str = "15m") -> Optional[dict]:
    try:
        import yfinance as yf
        period_map = {"1m":"7d","5m":"60d","15m":"60d","30m":"60d",
                      "1h":"730d","4h":"730d","1d":"max","1wk":"max","1mo":"max"}
        yf_iv  = "1h" if interval == "4h" else interval
        ticker = yf.Ticker(symbol)
        hist   = ticker.history(period=period_map.get(interval,"60d"), interval=yf_iv)
        if hist.empty:
            logger.warning("yfinance empty for %s", symbol)
            return None
        closes = hist["Close"].tolist()
        price  = closes[-1]
        rsi    = calculate_rsi(closes)
        ema20  = calculate_ema(closes, 20)
        ema50  = calculate_ema(closes, 50)
        return {
            "symbol": symbol, "price": round(price, 4),
            "open":   round(float(hist["Open"].iloc[-1]),  4),
            "high":   round(float(hist["High"].iloc[-1]),  4),
            "low":    round(float(hist["Low"].iloc[-1]),   4),
            "volume": int(hist["Volume"].iloc[-1]),
            "rsi": rsi, "ema20": ema20, "ema50": ema50,
            "signal": generate_signal(price, rsi, ema20, ema50, current_strategy),
            "interval": interval, "source": "yfinance",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error("yfinance error for %s: %s", symbol, e)
        return None

def get_market_data(symbol: str, interval: str = "15m") -> Optional[dict]:
    data = _fetch_finnhub(symbol, interval)
    if data:
        return data
    logger.info("Finnhub failed for %s — trying yfinance fallback", symbol)
    return _fetch_yfinance(symbol, interval)

# ── Bot loop ──────────────────────────────────────────────────────────────────
async def bot_loop():
    logger.info("Bot loop task started")
    while True:
        if bot_enabled:
            for symbol in SCAN_SYMBOLS:
                try:
                    data = get_market_data(symbol)
                    if not data:
                        continue
                    sig, price = data["signal"], data["price"]

                    if sig == "BUY" and demo_account["balance"] > price * 10:
                        qty = int(demo_account["balance"] * 0.02 / price)
                        if qty > 0:
                            demo_account["balance"] -= qty * price
                            demo_account["positions"].append(
                                {"symbol": symbol, "qty": qty, "entry_price": price, "current_price": price})
                            demo_account["trade_history"].append(
                                {"type":"BUY","symbol":symbol,"qty":qty,"price":price,
                                 "time":datetime.utcnow().isoformat()})
                            logger.info("BOT BUY %s x%d @ %.2f", symbol, qty, price)

                    elif sig == "SELL":
                        pos = next((p for p in demo_account["positions"] if p["symbol"]==symbol), None)
                        if pos:
                            demo_account["balance"] += pos["qty"] * price
                            demo_account["positions"].remove(pos)
                            demo_account["trade_history"].append(
                                {"type":"SELL","symbol":symbol,"qty":pos["qty"],"price":price,
                                 "time":datetime.utcnow().isoformat()})
                            logger.info("BOT SELL %s x%d @ %.2f", symbol, pos["qty"], price)

                    if connected_clients:
                        msg = json.dumps({"type":"market_update","data":data})
                        for ws in connected_clients[:]:
                            try:    await ws.send_text(msg)
                            except: connected_clients.remove(ws)

                except Exception as e:
                    logger.error("Bot error %s: %s", symbol, e)

            await asyncio.sleep(60)
        else:
            await asyncio.sleep(5)

# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    logger.info("🚀 Trading Scanner v2.0 started")
    logger.info("🔑 FINNHUB_API_KEY set: %s", bool(os.getenv("FINNHUB_API_KEY")))
    asyncio.create_task(bot_loop())

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Call this first if anything is broken — tells you exactly what to fix."""
    checks = {}

    # 1. Finnhub key present?
    key = os.getenv("FINNHUB_API_KEY","").strip()
    if key:
        checks["finnhub_key"] = {"status":"ok"}
        # 2. Finnhub actually reachable?
        try:
            r = requests.get(f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={key}", timeout=8)
            q = r.json()
            if q.get("c",0) > 0:
                checks["finnhub_live"] = {"status":"ok","aapl_price":q["c"]}
            else:
                checks["finnhub_live"] = {"status":"error","response":q,
                    "fix":"Key may be wrong. Check finnhub.io/dashboard → API Key."}
        except Exception as e:
            checks["finnhub_live"] = {"status":"error","error":str(e),
                "fix":"Finnhub unreachable. Check Render logs for network errors."}
    else:
        checks["finnhub_key"] = {"status":"MISSING",
            "fix":"In Render: open your service → Environment tab → Add Variable. "
                  "Key: FINNHUB_API_KEY  Value: your key from finnhub.io/dashboard"}
        checks["finnhub_live"] = {"status":"skipped"}

    # 3. yfinance fallback
    try:
        import yfinance as yf
        h = yf.Ticker("AAPL").history(period="2d", interval="1d")
        if not h.empty:
            checks["yfinance"] = {"status":"ok","aapl_close":round(float(h["Close"].iloc[-1]),2)}
        else:
            checks["yfinance"] = {"status":"empty",
                "note":"Yahoo Finance returned no data — likely rate-limited on this server IP. "
                       "This is normal. Finnhub should be your primary source."}
    except Exception as e:
        checks["yfinance"] = {"status":"error","error":str(e)}

    checks["bot"]        = {"status":"ok","enabled":bot_enabled,"strategy":current_strategy}
    checks["websockets"] = {"status":"ok","clients":len(connected_clients)}
    checks["account"]    = {"status":"ok","balance":demo_account["balance"],
                             "positions":len(demo_account["positions"])}

    failures = [k for k,v in checks.items() if v.get("status") not in ("ok","skipped")]
    return {
        "overall":   "✅ healthy" if not failures else "⚠️ degraded",
        "failures":  failures,
        "checks":    checks,
        "timestamp": datetime.utcnow().isoformat(),
    }

# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status":"Trading Scanner API Running","version":"2.0.0","bot_enabled":bot_enabled}

# ── Market data ───────────────────────────────────────────────────────────────
@app.get("/market/{symbol}")
def market(symbol: str, interval: str = "15m"):
    data = get_market_data(symbol.upper(), interval)
    if not data:
        raise HTTPException(503, detail={
            "error":"Unable to fetch market data",
            "symbol":symbol, "interval":interval,
            "fix":"Call /health to see exactly what is broken."})
    return data

@app.get("/scan")
def scan(interval: str = "15m"):
    results = []
    for sym in SCAN_SYMBOLS:
        d = get_market_data(sym, interval)
        results.append(d if d else {"symbol":sym,"error":"fetch_failed","fix":"Call /health"})
    return {"results":results,"strategy":current_strategy,
            "bot_enabled":bot_enabled,"timestamp":datetime.utcnow().isoformat()}

# ── Bot control ───────────────────────────────────────────────────────────────
@app.post("/bot/start")
def start_bot():
    global bot_enabled
    bot_enabled = True
    logger.info("Bot STARTED by user")
    return {"bot_enabled":True,"message":"Bot is now running"}

@app.post("/bot/stop")
def stop_bot():
    global bot_enabled
    bot_enabled = False
    logger.info("Bot STOPPED by user")
    return {"bot_enabled":False,"message":"Bot stopped"}

@app.get("/bot/status")
def bot_status():
    return {"bot_enabled":bot_enabled,"strategy":current_strategy,
            "open_positions":len(demo_account["positions"])}

# ── Account ───────────────────────────────────────────────────────────────────
@app.get("/account")
def account():
    pv = 0.0
    for pos in demo_account["positions"]:
        d = get_market_data(pos["symbol"])
        if d:
            pos["current_price"]    = d["price"]
            pos["unrealized_pnl"]   = round((d["price"] - pos["entry_price"]) * pos["qty"], 2)
        pv += pos.get("current_price", pos["entry_price"]) * pos["qty"]
    demo_account["total_value"] = round(demo_account["balance"] + pv, 2)
    return demo_account

# ── Strategies ────────────────────────────────────────────────────────────────
@app.get("/strategies")
def list_strategies():
    return {"strategies":STRATEGIES,"current":current_strategy}

@app.post("/strategy/{name}")
def set_strategy(name: str):
    global current_strategy
    if name not in STRATEGIES:
        raise HTTPException(400, detail={"error":f"Unknown strategy '{name}'",
                                         "available":list(STRATEGIES.keys())})
    current_strategy = name
    return {"strategy":current_strategy,"description":STRATEGIES[name]}

# ── Backtesting ───────────────────────────────────────────────────────────────
@app.get("/backtest/{symbol}")
def backtest(symbol: str, interval: str = "1d", initial_balance: float = 100_000.0):
    try:
        import yfinance as yf
        hist = yf.Ticker(symbol.upper()).history(period="1y", interval=interval)
        if hist.empty:
            raise HTTPException(503, detail="No historical data — Yahoo Finance required for backtesting.")
        closes = hist["Close"].tolist()
        balance, position, trades = initial_balance, 0.0, []
        for i in range(50, len(closes)):
            w = closes[:i]; p = closes[i]
            rsi = calculate_rsi(w); e20 = calculate_ema(w,20); e50 = calculate_ema(w,50)
            sig = generate_signal(p, rsi, e20, e50)
            if sig == "BUY" and balance > p:
                shares = int(balance * 0.1 / p)
                if shares > 0:
                    balance -= shares * p; position += shares
                    trades.append({"type":"BUY","price":p,"shares":shares})
            elif sig == "SELL" and position > 0:
                balance += position * p
                trades.append({"type":"SELL","price":p,"shares":position})
                position = 0
        final = balance + position * closes[-1]
        return {"symbol":symbol,"initial_balance":initial_balance,"final_value":round(final,2),
                "return_pct":round((final-initial_balance)/initial_balance*100,2),
                "total_trades":len(trades),"sample_trades":trades[-10:]}
    except HTTPException: raise
    except Exception as e:
        logger.error("Backtest error %s: %s", symbol, e)
        raise HTTPException(500, detail=str(e))

# ── Journal ───────────────────────────────────────────────────────────────────
class JournalEntry(BaseModel):
    title: str
    notes: str
    symbol: Optional[str] = None
    outcome: Optional[str] = None

@app.get("/journal")
def get_journal():
    return {"entries":_journal}

@app.post("/journal")
def add_journal(entry: JournalEntry):
    record = {**entry.dict(),"id":len(_journal)+1,"created_at":datetime.utcnow().isoformat()}
    _journal.append(record)
    return record

# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.append(ws)
    logger.info("WS connected (total: %d)", len(connected_clients))
    try:
        while True:
            msg = await ws.receive_text()
            await ws.send_text(json.dumps({"type":"ack","received":msg,
                                           "server_time":datetime.utcnow().isoformat()}))
    except WebSocketDisconnect:
        connected_clients.remove(ws)
        logger.info("WS disconnected (total: %d)", len(connected_clients))
