import numpy as np
import pandas as pd
import yfinance as yf
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
!pip install PyPortfolioOpt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import sample_cov
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

# USER INPUT 
user_risk_label = "High Risk"  # this would be changed based on KMeans output

# FD & PPF CALCULATORS 
def fd_return(principal, rate, years):
    return principal * ((1 + rate) ** years)

def ppf_return(principal, rate, years):
    return principal * ((1 + rate) ** years)

# SIP CALCULATOR
def sip_return(monthly_investment, rate, years):
    r = rate / 12
    n = years * 12
    return monthly_investment * (((1 + r) ** n - 1) / r) * (1 + r)

# Create model directory if not exists
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

# LSTM Model Builder (Train for each stock or load if saved)
def train_or_load_lstm(ticker):
    model_path = f"{model_dir}/lstm_{ticker}_model.h5"
    if os.path.exists(model_path):
        return load_model(model_path)

    df = yf.download(ticker, start="2019-01-01", end="2023-01-01")[['Close', 'Volume']]
    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close', 'Volume', 'Return']])

    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X_train, y_train = X[:int(len(X)*0.8)], y[:int(len(X)*0.8)]

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, callbacks=[EarlyStopping(patience=3)])
    model.save(model_path)
    return model



# Predict future return for a single stock 
def predict_future_return(ticker):
    df = yf.download(ticker, period="90d")[['Close', 'Volume']]
    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)

    if len(df) < 60:
        raise ValueError(f"Not enough data for {ticker}")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close', 'Volume', 'Return']])
    X_input = np.array([scaled[-60:]])

    model = train_or_load_lstm(ticker)
    pred_scaled = model.predict(X_input)
    pred_row = np.hstack((pred_scaled, np.zeros((1, scaled.shape[1] - 1))))
    pred_close = scaler.inverse_transform(pred_row)[0][0]

    current_price = float(df['Close'].iloc[-1])
    return (pred_close - current_price) / current_price


    # Evaluate a list of tickers and get top N performers 
def get_top_n_stocks(tickers, top_n=5):
    predictions = {}
    for ticker in tickers:
        try:
            predicted_return = predict_future_return(ticker)
            predictions[ticker] = predicted_return
            print(f"{ticker}: predicted return = {predicted_return:.2%}")
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    top_stocks = dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n])
    return top_stocks

# MVO Integration 
def run_mvo(top_stocks):
    mu = pd.Series(top_stocks)
    prices = yf.download(list(top_stocks.keys()), start="2023-01-01", end="2024-01-01")["Close"]
    S = sample_cov(prices)
    ef = EfficientFrontier(mu, S)
    ef.max_sharpe()
    weights = ef.clean_weights()
    print("\n💼 Optimized Portfolio Allocation:")
    for asset, weight in weights.items():
        print(f"{asset}: {weight*100:.2f}%")
    ef.portfolio_performance(verbose=True)
    
    # Main Logic Based on Risk
if user_risk_label == "Low Risk":
    print("\n📌 Suggested Instruments: PPF, Fixed Deposits, Bonds")
    print("FD Return (5 years @ 7%):", round(fd_return(100000, 0.07, 5), 2))
    print("PPF Return (15 years @ 7.1%):", round(ppf_return(100000, 0.071, 15), 2))

elif user_risk_label == "Medium Risk":
    print("\n📌 Suggested Instruments: Mutual Funds, Index Funds")
    print("SIP Return (5k/month @ 12% for 10 years):", round(sip_return(5000, 0.12, 10), 2))

elif user_risk_label == "High Risk":
    print("\n📌 High Risk Investor — Scanning Stocks with LSTM")
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NFLX", "NVDA", "INTC", "ORCL"]
    top_stocks = get_top_n_stocks(tickers, top_n=5)
    print("\nTop Stocks Based on Predicted Returns:")
    for stock, ret in top_stocks.items():
        print(f"{stock}: {ret:.2%}")
    run_mvo(top_stocks)