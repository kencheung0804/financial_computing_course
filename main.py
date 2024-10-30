import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import backtrader as bt
from datetime import datetime
from strategy import KMeansTradingStrategy
import os

data_dir = "data"

# Initialize an empty dictionary for the universe
universe = []
tickers = []

# Loop through each CSV file in the data directory
for filename in os.listdir(data_dir):
    if filename.endswith(".csv"):
        # Create the ticker name by removing '.csv' from the filename
        ticker = filename.replace(".csv", "")
        tickers.append(ticker)

        # Load the data and filter for 2010-2015
        data = pd.read_csv(
            os.path.join(data_dir, filename), index_col="Date", parse_dates=True
        ).loc["2010":"2015"]

        # Add the data to the universe dictionary with the ticker name as the key
        universe.append(data)
window_size = 5  # Last 5 days for features
forecast_size = 3  # Future 3 days for regression target

# Initialize list to store dataset
data_rows = []

for stock_data in universe:
    stock_data = stock_data[["Adj Close", "Volume"]]  # Select only Adj Close and Volume
    for i in range(window_size, len(stock_data) - forecast_size):
        # Get last 5 days of Adj Close and Volume
        last_5_days = stock_data.iloc[i - window_size : i]

        # Get future 3 days of Adj Close for trend prediction
        future_3_days = stock_data["Adj Close"].iloc[i : i + forecast_size].values

        # Flatten last 5 days' data and future 3 days
        features = np.hstack(
            [last_5_days["Adj Close"].values, last_5_days["Volume"].values]
        )
        target = future_3_days

        # Append the row with features and target
        data_rows.append((features, target))

# Convert to DataFrame for easier handling
features_df = pd.DataFrame(
    [row[0] for row in data_rows],
    columns=[f"AdjClose_{i}" for i in range(1, 6)]
    + [f"Volume_{i}" for i in range(1, 6)],
)
target_df = pd.DataFrame(
    [row[1] for row in data_rows],
    columns=["FutureAdjClose_1", "FutureAdjClose_2", "FutureAdjClose_3"],
)

# Normalize features for clustering
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(features_df.to_numpy())

# Step 1: Cluster the normalized data with K-means
k_clusters = 5
kmeans = KMeans(n_clusters=k_clusters, random_state=0)
kmeans.fit(normalized_features)

# Step 2: Classify clusters based on the trend of future 3 days
up_clusters = set()
for cluster in range(k_clusters):
    # Filter targets for each cluster
    cluster_indices = [i for i, label in enumerate(kmeans.labels_) if label == cluster]
    cluster_targets = target_df.iloc[cluster_indices].mean(axis=1)

    # Apply linear regression to determine if the cluster is "UP" or "DOWN"
    if len(cluster_targets) > 1:
        reg = LinearRegression().fit(
            np.arange(len(cluster_targets)).reshape(-1, 1), cluster_targets
        )
        slope = reg.coef_[0]
        if slope > 0:
            up_clusters.add(cluster)


def setup_cerebro_with_kmeans(kmeans, up_clusters, scaler):
    cerebro = bt.Cerebro()

    # Load your data files for each ticker

    for ticker in tickers:
        data = bt.feeds.YahooFinanceCSVData(
            dataname=f"data/{ticker}.csv",
            fromdate=datetime(2016, 1, 1),
            todate=datetime(2024, 12, 31),
            plot=False,
        )
        cerebro.adddata(data)

    # Add strategy with KMeans model, clusters, and scaler
    cerebro.addstrategy(
        KMeansTradingStrategy, kmeans=kmeans, up_clusters=up_clusters, scaler=scaler
    )

    return cerebro


# Assuming 'kmeans', 'up_clusters', and 'scaler' are already defined based on 2000-2005 data
cerebro = setup_cerebro_with_kmeans(kmeans, up_clusters, scaler)
cerebro.addobserver(bt.observers.Value)
cerebro.addobserver(bt.observers.DrawDown)
cerebro.addobserver(bt.observers.Cash)

cerebro.run(stdstats=False)
cerebro.plot()
