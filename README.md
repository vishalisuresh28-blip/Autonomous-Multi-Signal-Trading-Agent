# \# 🤖 Autonomous Multi-Signal Trading Agent

# 

# An end-to-end \*\*agentic AI trading system\*\* that integrates market data, NLP-based sentiment analysis, and machine learning models to generate \*\*Buy / Sell / Hold decisions\*\* with confidence scores.

# 

# \---

# 

# \## 🚀 Overview

# 

# This project implements an \*\*autonomous trading agent\*\* following a structured decision loop:

# 

# > \*\*Observe → Plan → Execute → Reflect → Decide → Act\*\*

# 

# The agent combines:

# \- 📊 Technical indicators

# \- 📰 News sentiment (FinBERT + VADER + TextBlob)

# \- 🤖 Machine learning models (Random Forest, LSTM)

# 

# to produce \*\*explainable and adaptive trading decisions\*\*.

# 

# \---

# 

# \## 🧠 Key Features

# 

# \### 🔹 Agentic Decision System

# \- Multi-step reasoning loop

# \- Dynamic signal aggregation

# \- Conflict detection and resolution

# \- Confidence scoring mechanism

# 

# \---

# 

# \### 🔹 Robust Data Pipeline

# \- 3-tier fallback system for stock data:

# &#x20; 1. Yahoo Finance API (direct HTTP)

# &#x20; 2. yfinance Ticker API

# &#x20; 3. yfinance download fallback

# \- Handles rate limits using:

# &#x20; - Session headers

# &#x20; - Optional caching (`requests\_cache`)

# &#x20; - Streamlit caching (1-hour TTL)

# 

# \---

# 

# \### 🔹 Sentiment Analysis (NLP)

# \- \*\*FinBERT\*\* (primary financial sentiment model)

# \- \*\*VADER + TextBlob\*\* (fallback)

# \- \*\*BART\*\* for news summarization

# \- Weighted sentiment blending for robustness

# 

# \---

# 

# \### 🔹 Machine Learning Models

# \- \*\*Random Forest\*\*

# &#x20; - Uses technical indicators: SMA, EMA, RSI, Returns

# \- \*\*LSTM Neural Network\*\*

# &#x20; - Time-series forecasting with sequence learning

# \- Performance metrics:

# &#x20; - R² Score

# &#x20; - MAE (Mean Absolute Error)

# 

# \---

# 

# \### 🔹 Intelligent Signal Fusion

# \- Converts model outputs into signals:

# &#x20; - +1 → BUY

# &#x20; - 0 → HOLD

# &#x20; - -1 → SELL

# \- Detects \*\*conflicting signals\*\*

# \- Dynamically adjusts weights to reduce noise

# 

# \---

# 

# \### 🔹 Explainability (Core Highlight)

# \- Full \*\*chain-of-thought reasoning log\*\*

# \- Transparent:

# &#x20; - Model outputs

# &#x20; - Signal votes

# &#x20; - Final decision logic

# 

# \---

# 

# \### 🔹 Backtesting Engine

# \- Simulates trading strategy over historical data

# \- Compares against \*\*Buy \& Hold benchmark\*\*

# \- Outputs:

# &#x20; - Portfolio value

# &#x20; - Returns

# &#x20; - Trade history

# &#x20; - Alpha

# 

# \---

# 

# \### 🔹 Interactive Dashboard

# Built using \*\*Streamlit\*\*:

# \- Company search (no ticker needed)

# \- Live agent execution

# \- Visualizations:

# &#x20; - Price charts

# &#x20; - Sentiment analysis

# &#x20; - Signal breakdown

# &#x20; - Backtest results

# 

# \---

# 

# 



