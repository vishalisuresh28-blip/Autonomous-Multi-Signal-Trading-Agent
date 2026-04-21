# 🤖 Autonomous Multi-Signal Trading Agent

An end-to-end agentic AI trading system that combines financial data, news sentiment, and machine learning models to generate Buy / Sell / Hold decisions with confidence scores and full reasoning transparency.

---

## Overview

This project implements an autonomous trading agent using a structured decision-making loop:

Observe → Plan → Execute → Reflect → Decide → Act

Instead of relying on a single model, the system combines multiple independent signals (market data, sentiment, ML forecasts) and intelligently fuses them into a final decision.

---

## How the System Works (Step-by-Step)

### 1. Observe (Data Collection)

The agent first gathers all required inputs:

- **Stock Price Data**
  - Fetches up to 5 years of historical data
  - Uses a 3-layer fallback system to avoid failures:
    - Direct Yahoo Finance API (fastest and most reliable)
    - yfinance Ticker API
    - yfinance download (backup)

- **Technical Indicators Computed**
  - SMA (20-day moving average)
  - EMA (20-day exponential moving average)
  - RSI (momentum indicator)
  - Daily returns

- **News Data**
  - Fetches latest headlines using Google News RSS
  - Extracts top 10 relevant articles

---

### 2. Plan (Strategy Selection)

The agent decides which models to run based on available data:

- Random Forest → always used  
- LSTM → only used if sufficient data (>= 200 rows)  

This avoids unnecessary computation and prevents unreliable predictions.

---

### 3. Execute (Model Processing)

#### 🔹 Sentiment Analysis

Each news headline is analyzed using:

- **FinBERT (primary model)**
  - Specialized for financial text
  - Outputs positive / negative / neutral scores

- **Fallback: VADER + TextBlob**
  - Used if FinBERT fails
  - Ensures robustness

- **Score Blending**
  - Final sentiment = weighted combination of models

- **Output**
  - Sentiment score (continuous)
  - Sentiment label (Bullish / Bearish / Neutral)
  - Signal (+1 / 0 / -1)

---

#### 🔹 News Summarization

- Uses BART model to generate a short summary of all headlines
- Helps interpret overall market sentiment quickly

---

#### 🔹 Random Forest Model

- Input features:
  - SMA, EMA, RSI, Returns

- What it does:
  - Learns relationships between indicators and price
  - Predicts next price

- Output:
  - Forecasted price
  - R² score (model accuracy)
  - MAE (error)
  - Trading signal (+1 / 0 / -1)

---

#### 🔹 LSTM Model

- Deep learning model for time-series prediction

- What it does:
  - Learns sequential price patterns
  - Uses last 60 days to predict next value

- Output:
  - Forecasted price
  - R² score
  - MAE
  - Trading signal

- Note:
  - Skipped if data is insufficient

---

### 4. Reflect (Conflict Detection)

The agent checks if signals disagree:

Example:
- Sentiment → BUY  
- RF → SELL  
- LSTM → HOLD  

If conflict is detected:
- The outlier signal is identified
- Its weight is reduced by 50%
- Remaining weight is redistributed

This improves decision stability.

---

### 5. Decide (Final Decision Logic)

The agent combines all signals using weighted scoring:

- Each signal contributes:
  - Sentiment (default 30%)
  - Random Forest (35%)
  - LSTM (35%)

- Decision rules:
  - Strong agreement → BUY / SELL
  - Weak signals → HOLD
  - Confidence score calculated from signal strength

---

### 6. Act (Output)

The agent outputs:

- Final decision (BUY / SELL / HOLD)
- Confidence score (%)
- Individual model signals
- Full reasoning trace (step-by-step logs)

---

## Backtesting Engine

The system includes a simulation module to evaluate performance:

- Simulates trading using model predictions
- Executes buy/sell decisions over time
- Tracks:
  - Portfolio value
  - Number of trades
  - Returns (%)
  - Comparison with Buy & Hold strategy
  - Alpha (excess return)

---

## Key Features Explained

### Agentic Design
Unlike traditional models, this system:
- Makes decisions in multiple steps
- Adapts to conflicting information
- Explains its reasoning

---

### Robust Data Handling
- Avoids API failures using fallback layers
- Handles rate limits using caching and session headers

---

### Explainability
- Every decision is logged
- Users can see:
  - Why a decision was made
  - Which model influenced it

---

### Multi-Model Intelligence
- Combines:
  - NLP (sentiment)
  - ML (Random Forest)
  - Deep Learning (LSTM)

---

## Installation

```bash
pip install -r requirements.txt