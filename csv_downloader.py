import yfinance as yf


def download_csv(ticker):
    # Download data from Yahoo Finance
    data = yf.download(ticker, start="2000-01-01", end="2024-12-31")
    # Save to CSV for future use
    data.columns = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
    data.to_csv(f"data/{ticker}.csv")


download_csv("KO")
