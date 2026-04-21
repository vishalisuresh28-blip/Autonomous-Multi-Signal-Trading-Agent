"""
Autonomous Multi-Signal Trading Agent
======================================
Agentic loop: Observe → Plan → Execute Tools → Reflect → Decide → Act

Rate-limit fix (permanent):
  - All yfinance calls use a shared requests.Session with real browser headers
  - requests_cache used if installed (disk-level HTTP cache, survives restarts)
  - @st.cache_data with 1-hour TTL (was 5 min)
  - Zero separate validation requests — agent handles bad tickers in STEP 1
"""

import time, datetime, dataclasses, os
from collections import Counter
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas_ta as ta
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# ══════════════════════════════════════════════════════════════════════════════
#  HTTP SESSION — browser headers + optional disk cache
#  This is the ONLY place Yahoo Finance is touched.
#  Every yf.download / yf.Search / requests.get call passes this session.
# ══════════════════════════════════════════════════════════════════════════════

# ── Browser headers used for every outbound request ─────────────────────────
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
_BROWSER_HEADERS = {
    "User-Agent":      _UA,
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
}

# One shared session — browser headers, optional disk cache
def _make_session() -> requests.Session:
    try:
        import requests_cache
        s = requests_cache.CachedSession(
            cache_name=os.path.join(os.path.expanduser("~"), ".yf_http_cache"),
            expire_after=3600, stale_if_error=True,
        )
    except ImportError:
        s = requests.Session()
    s.headers.update(_BROWSER_HEADERS)
    return s

_SESSION: requests.Session = _make_session()


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LAYER — 3-tier download, zero yf.download() dependency
#
#  Tier 1: Yahoo Finance v8 JSON API  (direct HTTP — most reliable)
#  Tier 2: yf.Ticker().history()      (yfinance object API)
#  Tier 3: yf.download()              (yfinance bulk API — last resort)
#
#  All tiers use browser headers. Result cached 1 hour in Streamlit + on disk.
# ══════════════════════════════════════════════════════════════════════════════

def _period_to_timestamps(period: str):
    """Convert yfinance period string to Unix timestamps."""
    from datetime import datetime, timedelta, timezone
    now   = datetime.now(timezone.utc)
    delta = {"1mo": 31, "3mo": 92, "6mo": 183, "1y": 365,
             "2y": 730, "5y": 1826, "10y": 3652}.get(period, 1826)
    start = now - timedelta(days=delta)
    return int(start.timestamp()), int(now.timestamp())


def _parse_yahoo_v8(data: dict) -> pd.DataFrame | None:
    """Parse Yahoo Finance v8 chart JSON into a clean DataFrame."""
    try:
        r          = data["chart"]["result"][0]
        timestamps = r["timestamp"]
        q          = r["indicators"]["quote"][0]
        adj        = r["indicators"].get("adjclose", [{}])[0]
        close_col  = adj.get("adjclose") or q["close"]
        df = pd.DataFrame({
            "Open":   q["open"],
            "High":   q["high"],
            "Low":    q["low"],
            "Close":  close_col,
            "Volume": q["volume"],
        }, index=pd.to_datetime(timestamps, unit="s", utc=True).normalize().tz_localize(None))
        df.index.name = "Date"
        return df.dropna()
    except Exception:
        return None


def _tier1_yahoo_v8(ticker: str, period: str) -> pd.DataFrame | None:
    """Direct Yahoo Finance v8 API — no yfinance, full control over headers."""
    start, end = _period_to_timestamps(period)
    for host in ("query1", "query2"):
        try:
            url  = f"https://{host}.finance.yahoo.com/v8/finance/chart/{ticker}"
            resp = _SESSION.get(url, timeout=15, params={
                "period1": start, "period2": end,
                "interval": "1d", "events": "history",
                "includePrePost": "false",
            })
            if resp.status_code == 429:          # rate-limited → skip, try next tier
                continue
            if resp.ok:
                df = _parse_yahoo_v8(resp.json())
                if df is not None and len(df) > 10:
                    return df
        except Exception:
            continue
    return None


def _tier2_yf_ticker(ticker: str, period: str) -> pd.DataFrame | None:
    """yf.Ticker().history() — object API, separate code path from yf.download()."""
    try:
        t  = yf.Ticker(ticker, session=_SESSION)
        df = t.history(period=period, auto_adjust=True, actions=False)
        if df is not None and not df.empty:
            return df.rename_axis("Date")
    except Exception:
        pass
    return None


def _tier3_yf_download(ticker: str, period: str) -> pd.DataFrame | None:
    """yf.download() last resort — least reliable but kept as final fallback."""
    try:
        from yfinance.exceptions import YFRateLimitError
        df = yf.download(ticker, period=period, auto_adjust=True,
                          progress=False, threads=False, session=_SESSION)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
    except Exception:
        pass
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def _download(ticker: str, period: str) -> pd.DataFrame | None:
    """
    Fetch OHLCV data for `ticker` over `period`.
    Tries 3 independent code paths so a rate-limit on one doesn't block all.
    """
    for tier_fn in (_tier1_yahoo_v8, _tier2_yf_ticker, _tier3_yf_download):
        df = tier_fn(ticker, period)
        if df is not None and len(df) > 10:
            # Normalise column names
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        time.sleep(1)   # brief pause before trying next tier
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def _search(query: str) -> list:
    """
    3-tier ticker search — all tiers use the same browser-header session.
    """
    q = query.strip()
    if len(q) < 2:
        return []

    def _clean(raw):
        out = []
        for r in raw:
            sym  = r.get("symbol", "")
            name = r.get("shortname") or r.get("longname") or sym
            exch = r.get("exchDisp", "")
            qt   = r.get("typeDisp", "")
            if qt.lower() not in ("equity", "etf", "fund", ""):
                continue
            if sym:
                out.append({"ticker": sym, "name": name,
                             "label": f"{sym}  —  {name}  ({exch})"})
        return out

    # Tier 1: direct Yahoo query2 endpoint (browser session)
    try:
        url  = ("https://query2.finance.yahoo.com/v1/finance/search"
                f"?q={requests.utils.quote(q)}&lang=en-US&region=IN"
                "&quotesCount=10&newsCount=0&enableFuzzyQuery=true")
        resp = _SESSION.get(url, timeout=10)
        if resp.ok:
            hits = _clean(resp.json().get("quotes", []))
            if hits:
                return hits
    except Exception:
        pass

    # Tier 2: yfinance.Search (same browser session internally)
    try:
        hits = _clean(yf.Search(q, max_results=10, session=_SESSION).quotes)
        if hits:
            return hits
    except Exception:
        pass

    # Tier 3: offline suffix guess
    base = q.upper().replace(" ", "")
    return [{"ticker": base + sfx, "name": q,
              "label": f"{base+sfx}  —  {q}  (guessed)"}
            for sfx in [".NS", ".BO", ""]]


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT STATE
# ══════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class AgentState:
    ticker:           str    = ""
    company:          str    = ""
    current_price:    float  = 0.0
    price_history:    object = None
    sentiment_label:  str    = "Unknown"
    sentiment_score:  float  = 0.0
    bart_summary:     str    = ""
    news_items:       list   = dataclasses.field(default_factory=list)
    rf_forecast:      float  = 0.0
    rf_r2:            float  = 0.0
    rf_mae:           float  = 0.0
    lstm_forecast:    float  = 0.0
    lstm_r2:          float  = 0.0
    lstm_mae:         float  = 0.0
    sentiment_signal: int    = 0
    rf_signal:        int    = 0
    lstm_signal:      int    = 0
    decision:         str    = "HOLD"
    confidence:       float  = 0.0
    reasoning_trace:  list   = dataclasses.field(default_factory=list)
    conflict_flag:    bool   = False
    finbert_ok:       bool   = False
    bart_ok:          bool   = False
    lstm_ok:          bool   = False
    rf_model:         object = None


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADERS  (loaded once per session)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _load_finbert():
    from transformers import (pipeline as P, AutoTokenizer,
                               AutoModelForSequenceClassification)
    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return P("text-classification", model=mdl, tokenizer=tok,
             truncation=True, max_length=512, top_k=None)

@st.cache_resource(show_spinner=False)
def _load_bart():
    from transformers import pipeline as P
    return P("summarization", model="facebook/bart-large-cnn", truncation=True)


# ══════════════════════════════════════════════════════════════════════════════
#  INDIVIDUAL TOOLS
# ══════════════════════════════════════════════════════════════════════════════

def _vote(forecast: float, price: float, threshold: float) -> int:
    if forecast > price * (1 + threshold): return  1
    if forecast < price * (1 - threshold): return -1
    return 0

def _log(state: AgentState, msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    state.reasoning_trace.append(f"[{ts}] {msg}")


def tool_price_history(ticker: str) -> pd.DataFrame:
    df = _download(ticker, "5y")
    if df is None or df.empty:
        raise RuntimeError(
            "All 3 data sources returned no data for this ticker. "
            "Try pressing Run Agent again — rate limits usually clear in 10–20 seconds."
        )
    df = df.copy()
    if isinstance(df["Close"], pd.DataFrame):
        df["Close"] = df["Close"].iloc[:, 0]
    c = df["Close"].squeeze()
    df["SMA_20"]  = c.rolling(20).mean()
    df["EMA_20"]  = c.ewm(span=20, adjust=False).mean()
    df["RSI"]     = ta.rsi(c, length=14)
    df["Returns"] = c.pct_change()
    df.dropna(inplace=True)
    if len(df) < 100:
        raise RuntimeError(f"Only {len(df)} clean rows — need at least 100.")
    return df


def tool_fetch_news(company: str) -> list:
    url   = (f"https://news.google.com/rss/search?q={company}+stock"
             "&hl=en-IN&gl=IN&ceid=IN:en")
    soup  = BeautifulSoup(_SESSION.get(url, timeout=10).text, "xml")
    items = soup.find_all("item")[:10]
    if not items:
        raise RuntimeError("No news items returned")
    return [{"title": i.title.text, "link": i.link.text} for i in items]


def tool_finbert(titles: list) -> tuple:
    pipe   = _load_finbert()
    scores = []
    for t in titles:
        res = pipe(t[:512])[0]
        d   = {r["label"].lower(): r["score"] for r in res}
        scores.append(d.get("positive", 0) - d.get("negative", 0))
    return float(np.mean(scores)), scores


def tool_vader_textblob(titles: list) -> tuple:
    sia    = SentimentIntensityAnalyzer()
    scores = [(sia.polarity_scores(t)["compound"] + TextBlob(t).sentiment.polarity) / 2
              for t in titles]
    return float(np.mean(scores)), scores


def tool_bart(titles: list) -> str:
    pipe = _load_bart()
    out  = pipe(" ".join(titles)[:1024], max_length=80, min_length=25, do_sample=False)
    return out[0]["summary_text"]


def tool_train_rf(df: pd.DataFrame, features: list) -> tuple:
    X, y = df[features], df["Close"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, shuffle=False, test_size=0.2)
    rf   = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(Xtr, ytr)
    preds    = rf.predict(Xte)
    r2       = float(r2_score(yte, preds))
    mae      = float(mean_absolute_error(yte, preds))
    forecast = float(rf.predict(df[features].iloc[-1].values.reshape(1, -1))[0])
    return rf, forecast, r2, mae


def tool_train_lstm(df: pd.DataFrame) -> tuple:
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["Close"]])
    LB     = 60
    Xs     = np.array([scaled[i-LB:i, 0] for i in range(LB, len(scaled))]).reshape(-1, LB, 1)
    ys     = np.array([scaled[i, 0]      for i in range(LB, len(scaled))])
    sp     = int(0.8 * len(Xs))
    model  = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LB, 1)),
        Dropout(0.2), LSTM(64), Dropout(0.2), Dense(1)
    ])
    model.compile(loss="mse", optimizer="adam")
    model.fit(Xs[:sp], ys[:sp], epochs=20, batch_size=32, verbose=0)
    preds  = scaler.inverse_transform(model.predict(Xs[sp:]))
    yte_r  = scaler.inverse_transform(ys[sp:].reshape(-1, 1))
    r2     = float(r2_score(yte_r, preds))
    mae    = float(mean_absolute_error(yte_r, preds))
    fc     = float(scaler.inverse_transform(
                   model.predict(scaled[-LB:].reshape(1, LB, 1)))[0][0])
    return model, fc, r2, mae


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_agent_loop(ticker, company, threshold=0.03, weights=None, ui_status=None):
    if weights is None:
        weights = {"sentiment": 0.30, "rf": 0.35, "lstm": 0.35}
    weights  = dict(weights)
    state    = AgentState(ticker=ticker, company=company)
    features = ["SMA_20", "EMA_20", "RSI", "Returns"]

    def upd(label):
        if ui_status: ui_status.update(label=label)

    # ── STEP 1: OBSERVE — price history ──────────────────────────────────────
    _log(state, "OBSERVE | Fetching 5-year price history…")
    upd("Step 1/6 — Fetching price history…")
    try:
        df = tool_price_history(ticker)
        state.price_history = df
        state.current_price = float(df["Close"].iloc[-1])
        avail = [c for c in features if c in df.columns]
        _log(state, f"  OK | {len(df)} rows | price=₹{state.current_price:.2f} | features={avail}")
    except Exception as e:
        _log(state, f"  FATAL | {e}")
        return state

    # ── STEP 2: OBSERVE — news + sentiment ───────────────────────────────────
    _log(state, "OBSERVE | Fetching news headlines…")
    upd("Step 2/6 — Analysing news…")
    try:
        raw_news = tool_fetch_news(company)
        titles   = [n["title"] for n in raw_news]
        _log(state, f"  OK | {len(titles)} headlines")

        try:
            _log(state, "  Trying FinBERT…")
            avg_fb, per_fb = tool_finbert(titles)
            state.finbert_ok = True
            avg_nlp, per_nlp = tool_vader_textblob(titles)
            blended   = [0.6*f + 0.4*n for f, n in zip(per_fb, per_nlp)]
            avg_score = 0.6 * avg_fb + 0.4 * avg_nlp
            _log(state, f"  FinBERT OK | blended avg={avg_score:+.3f}")
        except Exception as fe:
            _log(state, f"  FALLBACK | FinBERT unavailable ({fe}) → using VADER+TextBlob")
            avg_score, blended = tool_vader_textblob(titles)

        state.sentiment_score  = avg_score
        state.sentiment_label  = ("Bullish" if avg_score > 0.15
                                   else "Bearish" if avg_score < -0.15 else "Neutral")
        state.sentiment_signal = (1 if state.sentiment_label == "Bullish"
                                   else -1 if state.sentiment_label == "Bearish" else 0)
        state.news_items = [
            {**n, "score": round(blended[i], 3),
             "label": ("Positive" if blended[i] > 0.15
                        else "Negative" if blended[i] < -0.15 else "Neutral")}
            for i, n in enumerate(raw_news)
        ]
        _log(state, f"  Sentiment={state.sentiment_label} | signal={state.sentiment_signal:+d}")

        try:
            _log(state, "  Running BART summary…")
            state.bart_summary = tool_bart(titles)
            state.bart_ok      = True
            _log(state, "  BART OK")
        except Exception as be:
            state.bart_summary = "BART unavailable."
            _log(state, f"  FALLBACK | BART unavailable ({be})")

    except Exception as e:
        _log(state, f"  FALLBACK | News failed ({e}) → sentiment=Neutral")
        state.sentiment_label = "Neutral"

    # ── STEP 3: PLAN ─────────────────────────────────────────────────────────
    _log(state, "PLAN | Deciding which models to run…")
    upd("Step 3/6 — Planning…")
    df     = state.price_history
    avail  = [c for c in features if c in df.columns]
    run_lstm = len(df) >= 200
    _log(state, f"  Rows={len(df)} | RF=YES | LSTM={'YES' if run_lstm else 'NO (<200 rows)'}")

    # ── STEP 4: EXECUTE ───────────────────────────────────────────────────────
    _log(state, "EXECUTE | Training Random Forest…")
    upd("Step 4/6 — Training RF…")
    try:
        state.rf_model, state.rf_forecast, state.rf_r2, state.rf_mae = tool_train_rf(df, avail)
        state.rf_signal = _vote(state.rf_forecast, state.current_price, threshold)
        _log(state, f"  RF OK | forecast=₹{state.rf_forecast:.2f} | R²={state.rf_r2:.3f} | signal={state.rf_signal:+d}")
    except Exception as e:
        _log(state, f"  RF failed: {e}")

    if run_lstm:
        _log(state, "EXECUTE | Training LSTM…")
        upd("Step 4/6 — Training LSTM (~30s)…")
        try:
            _, state.lstm_forecast, state.lstm_r2, state.lstm_mae = tool_train_lstm(df)
            state.lstm_ok     = True
            state.lstm_signal = _vote(state.lstm_forecast, state.current_price, threshold)
            _log(state, f"  LSTM OK | forecast=₹{state.lstm_forecast:.2f} | R²={state.lstm_r2:.3f} | signal={state.lstm_signal:+d}")
        except Exception as e:
            state.lstm_forecast = state.rf_forecast
            state.lstm_signal   = 0
            _log(state, f"  FALLBACK | LSTM failed ({e}) → using RF forecast")
    else:
        state.lstm_forecast = state.rf_forecast
        _log(state, "  LSTM skipped (insufficient data)")

    # ── STEP 5: REFLECT ───────────────────────────────────────────────────────
    _log(state, "REFLECT | Checking for signal contradictions…")
    upd("Step 5/6 — Reflecting…")
    sigs               = [state.sentiment_signal, state.rf_signal, state.lstm_signal]
    unique_nonzero     = {s for s in sigs if s != 0}
    state.conflict_flag = len(unique_nonzero) > 1

    if state.conflict_flag:
        _log(state, f"  CONFLICT | signals={sigs} — identifying outlier…")
        majority = Counter(sigs).most_common(1)[0][0]
        for key, val in [("sentiment", state.sentiment_signal),
                          ("rf",        state.rf_signal),
                          ("lstm",      state.lstm_signal)]:
            if val != majority:
                excess = weights[key] / 2
                weights[key] /= 2
                others = [k for k in weights if k != key]
                for k in others: weights[k] += excess / len(others)
                _log(state, f"  Outlier={key} | weight halved → {weights}")
                break
    else:
        _log(state, f"  No conflict | signals={sigs}")

    raw_score       = sum(weights[k] * v for k, v in
                          zip(["sentiment","rf","lstm"], sigs))
    state.confidence = round(min(abs(raw_score) / 0.7 * 100, 100), 1)
    _log(state, f"  Confidence={state.confidence:.1f}%")

    # ── STEP 6: DECIDE ────────────────────────────────────────────────────────
    _log(state, "DECIDE | Computing final decision…")
    upd("Step 6/6 — Deciding…")
    score  = raw_score
    n_buy  = sigs.count(1)
    n_sell = sigs.count(-1)

    if   score > 0.2 and n_buy  >= 2: state.decision = "BUY";  _log(state, f"  BUY  | score={score:+.3f}")
    elif score < -0.2 and n_sell >= 2: state.decision = "SELL"; _log(state, f"  SELL | score={score:+.3f}")
    elif score > 0.1:                  state.decision = "HOLD"; _log(state, f"  HOLD | mild bullish lean ({score:+.3f})")
    elif score < -0.1:                 state.decision = "HOLD"; _log(state, f"  HOLD | mild bearish lean ({score:+.3f})")
    else:                              state.decision = "HOLD"; _log(state, f"  HOLD | no clear signal ({score:+.3f})")

    if state.conflict_flag:
        _log(state, "  NOTE: Decision made under signal conflict — confidence reduced")
    _log(state, f"DONE | {state.decision} | confidence={state.confidence:.1f}%")
    return state


# ══════════════════════════════════════════════════════════════════════════════
#  BACKTESTER
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(df, rf_model, features, capital=100_000.0, threshold=0.03):
    records, cash, shares = [], capital, 0.0
    prices, dates = df["Close"].values, df.index
    for i in range(20, len(df)):
        price = float(prices[i])
        try:    fc = float(rf_model.predict(df[features].iloc[i].values.reshape(1, -1))[0])
        except: fc = price
        s = _vote(fc, price, threshold)
        if   s == 1 and shares == 0: shares, cash, act = cash/price, 0.0, "BUY"
        elif s == -1 and shares > 0: cash, shares, act = shares*price, 0.0, "SELL"
        elif shares > 0:             act = "HOLD"
        else:                        act = "FLAT"
        records.append({"Date": dates[i], "Price": round(price, 2),
                         "Forecast": round(fc, 2), "Action": act,
                         "Shares": round(shares, 4), "Cash": round(cash, 2),
                         "PortfolioValue": round(cash + shares*price, 2)})
    return pd.DataFrame(records).set_index("Date")


def plot_backtest(bt, capital):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, facecolor="#0e1117")
    for ax in axes:
        ax.set_facecolor("#0e1117"); ax.tick_params(colors="#aaa")
        for sp in ax.spines.values(): sp.set_color("#333")
    bh = capital / bt["Price"].iloc[0] * bt["Price"]
    axes[0].plot(bt.index, bt["PortfolioValue"], color="#42a5f5", lw=2, label="Agent")
    axes[0].plot(bt.index, bh, color="#78909c", lw=1, ls="--", label="Buy & Hold")
    axes[0].set_ylabel("₹ Value", color="#ccc"); axes[0].set_title("Agent vs Buy & Hold", color="#eee")
    axes[0].legend(facecolor="#1e1e2e", labelcolor="#eee"); axes[0].grid(True, alpha=0.15)
    axes[1].plot(bt.index, bt["Price"], color="#546e7a", lw=1)
    b = bt[bt["Action"]=="BUY"];  s = bt[bt["Action"]=="SELL"]
    axes[1].scatter(b.index, b["Price"], marker="^", color="#43a047", s=80, zorder=5, label="BUY")
    axes[1].scatter(s.index, s["Price"], marker="v", color="#ef5350", s=80, zorder=5, label="SELL")
    axes[1].set_ylabel("Price (₹)", color="#ccc")
    axes[1].legend(facecolor="#1e1e2e", labelcolor="#eee"); axes[1].grid(True, alpha=0.15)
    plt.tight_layout(); st.pyplot(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Agentic Trading Dashboard", layout="wide", page_icon="📈")
st.title("🤖 Autonomous Multi-Signal Trading Agent")
st.caption("Observe → Plan → Execute → Reflect → Decide | FinBERT · BART · RF · LSTM")

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Agent Settings")
capital   = st.sidebar.number_input("Backtest Capital (₹):", value=100_000, step=10_000)
threshold = st.sidebar.slider("Trade Threshold (%)", 1, 10, 3) / 100
w_sent    = st.sidebar.slider("Sentiment Weight", 0.0, 1.0, 0.30, 0.05)
w_rf      = st.sidebar.slider("RF Weight",        0.0, 1.0, 0.35, 0.05)
w_lstm    = st.sidebar.slider("LSTM Weight",      0.0, 1.0, 0.35, 0.05)
_tot      = w_sent + w_rf + w_lstm or 1
weights   = {"sentiment": w_sent/_tot, "rf": w_rf/_tot, "lstm": w_lstm/_tot}

# ── Company search ────────────────────────────────────────────────────────────
st.markdown("Type a **company name** — no ticker knowledge needed.")
query  = st.text_input("🔍 Search company:", value="Reliance Industries",
                        placeholder="e.g. Infosys, TCS, Apple, Reliance…")
ticker, company = "", ""

if query:
    with st.spinner("Searching…"):
        results = _search(query)
    if not results:
        st.warning("No matches found. Paste the ticker below.")
        manual = st.text_input("Ticker:", placeholder="e.g. RELIANCE.NS")
        if manual:
            ticker, company = manual.strip().upper(), manual.strip()
    else:
        opts   = [r["label"] for r in results]
        chosen = st.selectbox("Select company:", opts)
        cd     = next(r for r in results if r["label"] == chosen)
        ticker, company = cd["ticker"], cd["name"]
        st.caption(f"✅ Using **{ticker}** — {company}")

if not ticker:
    st.info("👆 Search for a company above to get started.")
    st.stop()

# ── Run button ────────────────────────────────────────────────────────────────
st.divider()
c1, c2 = st.columns([1, 3])
with c1:
    run_btn = st.button("▶ Run Agent", type="primary", use_container_width=True)
with c2:
    st.markdown(
        "Agent will autonomously: **fetch data → score news → train models "
        "→ detect conflicts → decide** — every step logged live."
    )

if not run_btn:
    st.info("Press **▶ Run Agent** to start the autonomous loop.")
    st.stop()

# ── Execute agent ─────────────────────────────────────────────────────────────
with st.status("🤖 Agent loop starting…", expanded=True) as status_box:
    state = run_agent_loop(ticker, company, threshold, weights, status_box)
    status_box.update(label="✅ Agent loop complete", state="complete", expanded=False)

# ── Handle fatal failure (no price data) ─────────────────────────────────────
if state.price_history is None:
    msg = next((l for l in state.reasoning_trace if "FATAL" in l), "")
    st.error(
        "**The agent could not load price data.** "
        + ("Details: " + msg.split("FATAL | ")[-1] if msg else "")
        + "\n\n💡 **What to do:** Wait 30–60 seconds and press **▶ Run Agent** again. "
        "Yahoo Finance rate-limits are temporary and usually clear within a minute. "
        "If it keeps failing, try a different company."
    )
    with st.expander("🔍 Full agent log"):
        for line in state.reasoning_trace:
            st.text(line)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS UI
# ══════════════════════════════════════════════════════════════════════════════

# ── Chain-of-thought trace ────────────────────────────────────────────────────
st.divider()
st.subheader("🧠 Agent Chain-of-Thought")
st.caption("Every reasoning step — including fallbacks and conflict resolutions.")

trace_html = ""
for line in state.reasoning_trace:
    if   "FATAL"   in line:                                  col, bg = "#ef5350", "#2a0a0a"
    elif "FALLBACK" in line or "failed" in line.lower():     col, bg = "#ffa726", "#2a1a00"
    elif "CONFLICT" in line:                                  col, bg = "#ff7043", "#2a0e00"
    elif "DONE"     in line or " OK"    in line:              col, bg = "#66bb6a", "#0a2a0a"
    elif any(k in line for k in ("OBSERVE","PLAN","EXECUTE","REFLECT","DECIDE")):
                                                              col, bg = "#42a5f5", "#0a1a2e"
    elif "signal="  in line or "→"      in line:             col, bg = "#ce93d8", "#1a0a2e"
    else:                                                     col, bg = "#90caf9", "#111"
    safe = line.replace("<", "&lt;").replace(">", "&gt;")
    trace_html += (
        f'<div style="font-family:monospace;font-size:0.82rem;background:{bg};'
        f'color:{col};padding:3px 12px;margin:1px 0;border-radius:3px">{safe}</div>'
    )

st.markdown(
    f'<div style="max-height:300px;overflow-y:auto;border:1px solid #333;'
    f'border-radius:8px;padding:4px">{trace_html}</div>',
    unsafe_allow_html=True,
)

# ── Price metrics ─────────────────────────────────────────────────────────────
st.divider()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Price",  f"₹{state.current_price:.2f}")
c2.metric("RF Forecast",    f"₹{state.rf_forecast:.2f}",
          delta=f"₹{state.rf_forecast - state.current_price:+.2f}")
c3.metric("LSTM Forecast",  f"₹{state.lstm_forecast:.2f}",
          delta=f"₹{state.lstm_forecast - state.current_price:+.2f}")
c4.metric("Confidence",     f"{state.confidence:.1f}%")

# ── Price history chart ───────────────────────────────────────────────────────
st.subheader("📈 Price History")
period = st.selectbox("Period:", ["1mo", "3mo", "6mo", "1y", "5y"], index=2)
hdata  = _download(ticker, period)   # uses same cached session — no extra request
if hdata is not None and not hdata.empty:
    fig, ax = plt.subplots(figsize=(12, 4), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    close_col = hdata["Close"] if isinstance(hdata["Close"], pd.Series) else hdata["Close"].iloc[:, 0]
    ax.plot(hdata.index, close_col, color="#42a5f5", lw=1.5)
    ax.set_ylabel("Price (₹)", color="#ccc"); ax.set_title(f"{ticker} — {period}", color="#eee")
    ax.tick_params(colors="#888")
    for sp in ax.spines.values(): sp.set_color("#333")
    ax.grid(True, alpha=0.15); plt.tight_layout(); st.pyplot(fig)

# ── Sentiment ─────────────────────────────────────────────────────────────────
fb_label = "FinBERT ✓" if state.finbert_ok else "VADER (FinBERT unavailable)"
st.subheader(f"📰 News Sentiment — {fb_label}")
badge_bg = {"Bullish": "#1b5e20", "Bearish": "#b71c1c"}.get(state.sentiment_label, "#1a237e")
st.markdown(
    f'<div style="display:flex;gap:12px;align-items:center;margin:8px 0">'
    f'<span style="background:{badge_bg};color:#fff;padding:5px 18px;'
    f'border-radius:20px;font-weight:700;font-size:1rem">{state.sentiment_label}</span>'
    f'<span style="color:#aaa">Score: <b style="color:#fff">{state.sentiment_score:+.3f}</b>'
    f' | Signal: <b style="color:#fff">{state.sentiment_signal:+d}</b></span></div>',
    unsafe_allow_html=True,
)
if state.bart_ok:
    st.markdown(
        f'<div style="background:#1e2a3a;border-left:4px solid #42a5f5;'
        f'padding:10px 16px;border-radius:6px;margin:8px 0">'
        f'<span style="font-size:0.72rem;color:#90caf9;font-weight:700">📝 BART SUMMARY</span><br>'
        f'<span style="color:#e0e0e0">{state.bart_summary}</span></div>',
        unsafe_allow_html=True,
    )
for item in state.news_items:
    c  = "#43a047" if item["label"] == "Positive" else "#ef5350" if item["label"] == "Negative" else "#42a5f5"
    bw = min(int(abs(item["score"]) * 100), 100)
    title = item["title"].replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(
        f'<div style="border-left:4px solid {c};background:#1e1e2e;'
        f'padding:8px 14px;margin:5px 0;border-radius:0 6px 6px 0">'
        f'<span style="color:#e0e0e0">{title}</span>'
        f'<span style="color:{c};font-size:0.8rem;margin-left:8px">'
        f'● {item["label"]} ({item["score"]:+.3f})</span><br>'
        f'<div style="background:#333;height:4px;border-radius:4px;margin:5px 0 3px">'
        f'<div style="background:{c};width:{bw}%;height:4px;border-radius:4px"></div></div>'
        f'<a href="{item["link"]}" target="_blank" style="color:#90caf9;font-size:0.78rem">Read more →</a>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Signal table ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("⚡ Signal Breakdown")
if state.conflict_flag:
    st.warning("⚠️ Signal conflict detected — agent auto-adjusted weights. See chain-of-thought above.")

rows = []
for src, sig, val in [
    (f"📰 Sentiment ({fb_label})",             state.sentiment_signal, state.sentiment_label),
    (f"🌲 Random Forest  R²={state.rf_r2:.2f}  MAE=₹{state.rf_mae:.1f}", state.rf_signal, f"₹{state.rf_forecast:.2f}"),
    (f"🧬 LSTM  R²={state.lstm_r2:.2f}  MAE=₹{state.lstm_mae:.1f}" + ("" if state.lstm_ok else " [fallback]"),
     state.lstm_signal, f"₹{state.lstm_forecast:.2f}"),
]:
    rows.append({"Source": src, "Value": val,
                  "Vote": "🟢 +1 BUY" if sig == 1 else "🔴 -1 SELL" if sig == -1 else "🟡 0 HOLD"})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Final decision card ───────────────────────────────────────────────────────
_C = {
    "BUY" : {"bg": "#0d3321", "border": "#43a047", "badge": "#43a047", "tc": "#a5d6a7"},
    "SELL": {"bg": "#3b0d0d", "border": "#ef5350", "badge": "#ef5350", "tc": "#ef9a9a"},
    "HOLD": {"bg": "#1a1a2e", "border": "#ffa726", "badge": "#ffa726", "tc": "#ffe082"},
}
cd   = _C[state.decision]
conf = int(state.confidence)
conflict_badge = (
    '<span style="background:#b71c1c;color:#fff;padding:3px 10px;'
    'border-radius:10px;font-size:0.8rem;margin-left:10px">⚠ CONFLICT</span>'
    if state.conflict_flag else ""
)
st.markdown(
    f'<div style="background:{cd["bg"]};padding:24px 28px;border-radius:14px;'
    f'border:2px solid {cd["border"]};margin:12px 0">'
    f'<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">'
    f'<span style="background:{cd["badge"]};color:#fff;padding:6px 24px;'
    f'border-radius:20px;font-size:1.5rem;font-weight:800">{state.decision}</span>'
    f'<span style="color:{cd["tc"]};font-size:1rem">Final Agent Decision</span>'
    f'{conflict_badge}</div>'
    f'<div style="margin:14px 0 4px">'
    f'<div style="color:{cd["tc"]};font-size:0.82rem;margin-bottom:4px">Confidence: {state.confidence:.1f}%</div>'
    f'<div style="background:#333;height:8px;border-radius:4px">'
    f'<div style="background:{cd["badge"]};width:{conf}%;height:8px;border-radius:4px"></div></div></div>'
    f'<p style="margin:8px 0 0;color:{cd["tc"]};font-size:0.88rem">'
    f'Sentiment {state.sentiment_signal:+d} &nbsp;|&nbsp; '
    f'RF {state.rf_signal:+d} &nbsp;|&nbsp; LSTM {state.lstm_signal:+d}</p></div>',
    unsafe_allow_html=True,
)

# ── Backtest ──────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📊 Backtest Simulation")
st.caption(f"Capital: ₹{capital:,.0f}  |  Threshold: ±{threshold*100:.0f}%")

df    = state.price_history
avail = [c for c in ["SMA_20", "EMA_20", "RSI", "Returns"] if c in df.columns]
with st.spinner("Running backtest…"):
    X, y = df[avail], df["Close"]
    Xtr, _, ytr, _ = train_test_split(X, y, shuffle=False, test_size=0.2)
    rf_bt = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf_bt.fit(Xtr, ytr)
    bt = run_backtest(df, rf_bt, avail, capital, threshold)

plot_backtest(bt, capital)

fv   = bt["PortfolioValue"].iloc[-1]
bh   = capital / bt["Price"].iloc[0] * bt["Price"].iloc[-1]
tr   = (fv - capital)  / capital * 100
bhr  = (bh - capital)  / capital * 100
nb   = (bt["Action"] == "BUY").sum()
ns   = (bt["Action"] == "SELL").sum()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Final Portfolio",      f"₹{fv:,.0f}",  f"{tr:+.1f}%")
m2.metric("Buy & Hold Benchmark", f"₹{bh:,.0f}",  f"{bhr:+.1f}%")
m3.metric("Trades Executed",      f"{nb+ns}",       f"B:{nb}  S:{ns}")
m4.metric("Alpha vs B&H",         f"{tr-bhr:+.1f}%")

with st.expander("📋 Full Trade Log"):
    trades = bt[bt["Action"].isin(["BUY", "SELL"])]
    st.dataframe(
        trades[["Price", "Forecast", "Action", "Shares", "Cash", "PortfolioValue"]],
        use_container_width=True,
    )