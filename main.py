import backtrader
import yfinance as yf
import pandas as pd


# Load each CSV file into a DataFrame
msft_data = pd.read_csv("data/MSFT.csv", index_col="Date", parse_dates=True)
aapl_data = pd.read_csv("data/AAPL.csv", index_col="Date", parse_dates=True)
