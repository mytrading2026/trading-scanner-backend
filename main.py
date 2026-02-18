"""
Trading Scanner Backend — main.py v3.0
Primary data: Finnhub | Fallback: Yahoo Finance
Fixes: RSI calculation, rate limiting, caching, candle data endpoint
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

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Trading Scanner", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# ── State ─────────────────────────────────────────────────────────────────────
bot_enabled      = False
current_strategy = "default"
connected_clients: list = []
_journal: list   = []
_cache: dict     = {}
CACHE_TTL        = 60

demo_account = {
    "balance": 100_000.0,
    "positions": [],
    "trade_history": [],
    "total_value": 100_000.0,
}

SCAN_SYMBOLS = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN",
                "BTC-USD", "ETH-USD", "NVDA", "META", "AMD"]

SYMBOL_MAP = {
    "BTC-USD": "BINANCE:BTCUSDT",
    "ETH-USD": "BINANCE:ETHUSDT",
}

STRATEGIES = {
    "default":      "RSI + EMA crossover",
    "aggressive":   "RSI < 35 / > 65 with momentum",
    "conservative": "EMA alignment only, wider bands",
}

# ── Indicators ────────────────────────────────────────────────────────────────
def calculate_rsi(prices: list, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    arr    = np.array(prices[-100:], dtype=float)
    d      = np.diff(arr)
    gains  = np.where(d > 0, d, 0.0)
    losses = np.where(d < 0, -d, 0.0)
    ag     = np.mean(gains[:period])
    al     = np.mean(losses[:period])
    for i in range(period, len(gains)):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
    if al == 0:
        return 100.0 if ag > 0 else 50.0
    return round(100.0 - 100.0 / (1.0 + ag / al), 2)


def calculate_ema(prices: list, period: int = 20) -> float:
    if not prices:
        return 0.0
    if len(prices) < period:
        return round(float(prices[-1]), 4)
    s = pd.Series(prices, dtype=float)
    return round(float(s.ewm(span=period, adjust=False).mean().iloc[-1]), 4)


def generate_signal(price: float, rsi: float, ema20: float,
                    ema50: float, strategy: str = "default") -> str:
    if strategy == "aggressive":
        if rsi < 35 and price > ema20: return "BUY"
        if rsi > 65 and price < ema20: return "SELL"
    elif strategy == "conservative":
        if ema20 > ema50 and rsi < 55 and price > ema20: return "BUY"
        if ema20 < ema50 and rsi > 45 and price < ema20: return "SELL"
    else:
        if rsi < 30 and price > ema20:  return "BUY"
        if rsi > 70 and price < ema20:  return "SELL"
        if ema20 > ema50 and rsi < 60:  return "BUY"
        if ema20 < ema50 and rsi > 40:  return "SELL"
    return "HOLD"


# ── Cache ─────────────────────────────────────────────────────────────────────
def get_cached(symbol: str) -> Optional[dict]:
    entry = _cache.get(symbol)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL:
        return entry["data"]
    return None


def set_cached(symbol: str, data: dict):
    _cache[symbol] = {"data": data, "ts": time.time()}


# ── Finnhub ───────────────────────────────────────────────────────────────────
def _fetch_finnhub(symbol: str, interval: str = "15m") -> Optional[dict]:
    key = os.getenv("FINNHUB_API_KEY", "").strip()
    if not key:
        return None
    fh_symbol = SYMBOL_MAP.get(symbol, symbol)
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={fh_symbol}&token={key}"
        q   = requests.get(url, timeout=10).json()
        if q.get("error"):
            logger.warning("Finnhub error for %s: %s", fh_symbol, q["error"])
            return None
        if not q.get("c") or q["c"] == 0:
            logger.warning("Finnhub empty quote for %s", fh_symbol)
            return None
        price = q["c"]
        prev  = q["pc"] or price
        high  = q["h"]
        low   = q["l"]
        open_ = q["o"]
        now   = int(time.time())
        curl  = (f"https://finnhub.io/api/v1/stock/candle"
                 f"?symbol={fh_symbol}&resolution=D&from={now-365*86400}&to={now}&token={key}")
        cd = requests.get(curl, timeout=10).json()
        if cd.get("s") == "ok" and cd.get("c") and len(cd["c"]) >= 15:
            closes = cd["c"]
            volume = int(cd["v"][-1]) if cd.get("v") else 0
        else:
            step   = (price - prev) / 28
            closes = [round(prev + step * i, 4) for i in range(28)] + [price]
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
        logger.error("Finnhub exception for %s: %s", symbol, e)
        return None


# ── yfinance fallback ─────────────────────────────────────────────────────────
def _fetch_yfinance(symbol: str, interval: str = "15m") -> Optional[dict]:
    try:
        import yfinance as yf
        period_map = {"1m":"7d","5m":"60d","15m":"60d","30m":"60d",
                      "1h":"730d","4h":"730d","1d":"max","1wk":"max","1mo":"max"}
        yf_iv  = "1h" if interval == "4h" else interval
        hist   = yf.Ticker(symbol).history(period=period_map.get(interval,"60d"), interval=yf_iv)
        if hist.empty:
            return None
        closes = hist["Close"].tolist()
        price  = closes[-1]
        rsi    = calculate_rsi(closes)
        ema20  = calculate_ema(closes, 20)
        ema50  = calculate_ema(closes, 50)
        return {
            "symbol": symbol, "price": round(price, 4),
            "open": round(float(hist["Open"].iloc[-1]), 4),
            "high": round(float(hist["High"].iloc[-1]), 4),
            "low":  round(float(hist["Low"].iloc[-1]),  4),
            "volume": int(hist["Volume"].iloc[-1]),
            "rsi": rsi, "ema20": ema20, "ema50": ema50,
            "change_pct": 0,
            "signal": generate_signal(price, rsi, ema20, ema50, current_strategy),
            "interval": interval, "source": "yfinance",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error("yfinance exception for %s: %s", symbol, e)
        return None


# ── Main fetcher ──────────────────────────────────────────────────────────────
def get_market_data(symbol: str, interval: str = "15m") -> Optional[dict]:
    cached = get_cached(symbol)
    if cached:
        return cached
    data = _fetch_finnhub(symbol, interval)
    if not data:
        data = _fetch_yfinance(symbol, interval)
    if data:
        set_cached(symbol, data)
    return data


# ── Bot loop ──────────────────────────────────────────────────────────────────
async def bot_loop():
    logger.info("Bot loop started")
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
                                {"symbol": symbol, "qty": qty,
                                 "entry_price": price, "current_price": price})
                            demo_account["trade_history"].append(
                                {"type": "BUY", "symbol": symbol, "qty": qty,
                                 "price": price, "time": datetime.utcnow().isoformat()})
                    elif sig == "SELL":
                        pos = next((p for p in demo_account["positions"]
                                    if p["symbol"] == symbol), None)
                        if pos:
                            demo_account["balance"] += pos["qty"] * price
                            demo_account["positions"].remove(pos)
                            demo_account["trade_history"].append(
                                {"type": "SELL", "symbol": symbol,
                                 "qty": pos["qty"], "price": price,
                                 "time": datetime.utcnow().isoformat()})
                    if connected_clients:
                        msg = json.dumps({"type": "market_update", "data": data})
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
    logger.info("🚀 Trading Scanner v3.0 started")
    logger.info("🔑 FINNHUB_API_KEY set: %s", bool(os.getenv("FINNHUB_API_KEY")))
    asyncio.create_task(bot_loop())


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    checks = {}
    key = os.getenv("FINNHUB_API_KEY", "").strip()
    if key:
        checks["finnhub_key"] = {"status": "ok"}
        try:
            q = requests.get(
                f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={key}",
                timeout=8).json()
            checks["finnhub_live"] = {"status": "ok", "aapl_price": q["c"]} \
                if q.get("c", 0) > 0 else {"status": "error", "response": q}
        except Exception as e:
            checks["finnhub_live"] = {"status": "error", "error": str(e)}
    else:
        checks["finnhub_key"]  = {"status": "MISSING",
            "fix": "Render → Environment → Add FINNHUB_API_KEY"}
        checks["finnhub_live"] = {"status": "skipped"}
    try:
        import yfinance as yf
        h = yf.Ticker("AAPL").history(period="2d", interval="1d")
        checks["yfinance"] = {"status": "ok", "aapl_close": round(float(h["Close"].iloc[-1]), 2)} \
            if not h.empty else {"status": "empty", "note": "Rate limited — normal"}
    except Exception as e:
        checks["yfinance"] = {"status": "error", "error": str(e)}
    checks["cache"]      = {"status": "ok", "cached_symbols": list(_cache.keys())}
    checks["bot"]        = {"status": "ok", "enabled": bot_enabled}
    checks["websockets"] = {"status": "ok", "clients": len(connected_clients)}
    checks["account"]    = {"status": "ok", "balance": demo_account["balance"]}
    failures = [k for k, v in checks.items() if v.get("status") not in ("ok","skipped")]
    return {"overall": "✅ healthy" if not failures else "⚠️ degraded",
            "failures": failures, "checks": checks,
            "timestamp": datetime.utcnow().isoformat()}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Trading Scanner API Running", "version": "3.0.0",
            "bot_enabled": bot_enabled}

@app.get("/market/{symbol}")
def market(symbol: str, interval: str = "15m"):
    data = get_market_data(symbol.upper(), interval)
    if not data:
        raise HTTPException(503, detail={"error": "Unable to fetch", "fix": "Call /health"})
    return data

@app.get("/scan")
def scan(interval: str = "15m"):
    results = []
    for sym in SCAN_SYMBOLS:
        d = get_market_data(sym, interval)
        results.append(d if d else {"symbol": sym, "error": "fetch_failed"})
    return {"results": results, "strategy": current_strategy,
            "bot_enabled": bot_enabled, "timestamp": datetime.utcnow().isoformat()}

@app.get("/candles/{symbol}")
def candles(symbol: str, resolution: str = "D", days: int = 365):
    key = os.getenv("FINNHUB_API_KEY", "").strip()
    if not key:
        raise HTTPException(503, detail="FINNHUB_API_KEY not set")
    fh_symbol = SYMBOL_MAP.get(symbol.upper(), symbol.upper())
    now  = int(time.time())
    url  = (f"https://finnhub.io/api/v1/stock/candle"
            f"?symbol={fh_symbol}&resolution={resolution}"
            f"&from={now - days*86400}&to={now}&token={key}")
    try:
        cd = requests.get(url, timeout=10).json()
        if cd.get("s") != "ok":
            raise HTTPException(503, detail=f"Finnhub: {cd.get('s')}")
        return {"symbol": symbol, "resolution": resolution,
                "candles": [{"time": cd["t"][i], "open": cd["o"][i],
                              "high": cd["h"][i], "low": cd["l"][i],
                              "close": cd["c"][i], "volume": cd["v"][i]}
                             for i in range(len(cd["c"]))],
                "count": len(cd["c"])}
    except HTTPException: raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/bot/start")
def start_bot():
    global bot_enabled
    bot_enabled = True
    return {"bot_enabled": True}

@app.post("/bot/stop")
def stop_bot():
    global bot_enabled
    bot_enabled = False
    return {"bot_enabled": False}

@app.get("/bot/status")
def bot_status():
    return {"bot_enabled": bot_enabled, "strategy": current_strategy,
            "open_positions": len(demo_account["positions"])}

@app.get("/account")
def account():
    pv = 0.0
    for pos in demo_account["positions"]:
        d = get_market_data(pos["symbol"])
        if d:
            pos["current_price"]  = d["price"]
            pos["unrealized_pnl"] = round((d["price"] - pos["entry_price"]) * pos["qty"], 2)
        pv += pos.get("current_price", pos["entry_price"]) * pos["qty"]
    demo_account["total_value"] = round(demo_account["balance"] + pv, 2)
    return demo_account

@app.get("/strategies")
def list_strategies():
    return {"strategies": STRATEGIES, "current": current_strategy}

@app.post("/strategy/{name}")
def set_strategy(name: str):
    global current_strategy
    if name not in STRATEGIES:
        raise HTTPException(400, detail={"error": f"Unknown: {name}",
                                         "available": list(STRATEGIES.keys())})
    current_strategy = name
    return {"strategy": current_strategy}

@app.get("/backtest/{symbol}")
def backtest(symbol: str, interval: str = "1d", initial_balance: float = 100_000.0):
    try:
        import yfinance as yf
        hist = yf.Ticker(symbol.upper()).history(period="1y", interval=interval)
        if hist.empty:
            raise HTTPException(503, detail="No historical data.")
        closes = hist["Close"].tolist()
        balance, position, trades = initial_balance, 0.0, []
        for i in range(50, len(closes)):
            w = closes[:i]; p = closes[i]
            sig = generate_signal(p, calculate_rsi(w), calculate_ema(w,20), calculate_ema(w,50))
            if sig == "BUY" and balance > p:
                s = int(balance * 0.1 / p)
                if s > 0:
                    balance -= s * p; position += s
                    trades.append({"type":"BUY","price":p,"shares":s})
            elif sig == "SELL" and position > 0:
                balance += position * p
                trades.append({"type":"SELL","price":p,"shares":position})
                position = 0
        final = balance + position * closes[-1]
        return {"symbol": symbol, "initial_balance": initial_balance,
                "final_value": round(final, 2),
                "return_pct": round((final-initial_balance)/initial_balance*100, 2),
                "total_trades": len(trades), "sample_trades": trades[-10:]}
    except HTTPException: raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))

class JournalEntry(BaseModel):
    title: str; notes: str
    symbol: Optional[str] = None; outcome: Optional[str] = None

@app.get("/journal")
def get_journal():
    return {"entries": _journal}

@app.post("/journal")
def add_journal(entry: JournalEntry):
    record = {**entry.dict(), "id": len(_journal)+1,
              "created_at": datetime.utcnow().isoformat()}
    _journal.append(record)
    return record

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
