import yfinance as yf
import pandas as pd
import quantstats as qs


def download_csv(ticker):
    # Download data from Yahoo Finance
    data = yf.download(ticker, start="2000-01-01", end="2024-12-31")
    # Save to CSV for future use
    data.columns = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
    data.to_csv(f"data/{ticker}.csv")


def get_period_price(symbol, start, end):
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(start=start, end=end)
    return todays_data


def download_hsi_csv(ticker):
    # Download data from Yahoo Finance
    data = yf.download(ticker, start="2000-01-01", end="2024-12-31")
    # Save to CSV for future use
    data.columns = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
    data.to_csv(f"data_hsi/{ticker}.csv")


for t in ["0700.HK", "9988.HK", "3690.HK", "1211.HK", "2382.HK", "2800.HK"]:
    download_hsi_csv(t)
