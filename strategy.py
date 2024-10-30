import backtrader as bt
import numpy as np


class KMeansTradingStrategy(bt.Strategy):
    params = (("lookback", 5), ("k_clusters", 5))

    def __init__(self, kmeans, up_clusters, scaler):
        self.kmeans = kmeans
        self.up_clusters = up_clusters
        self.scaler = scaler

    def get_last_n_days(self, data, n):
        # Retrieve the last n days of Adj Close and Volume for a data feed
        return np.hstack((data.close.get(size=n), data.volume.get(size=n)))

    def next(self):
        buy_signals = []

        # Check each stock in the strategy
        for i, data in enumerate(self.datas):
            # Get last n days of Adj Close and Volume
            last_n_days = self.get_last_n_days(data, self.params.lookback)

            if len(last_n_days) < self.params.lookback * 2:
                # Not enough data points yet, continue to next iteration
                continue

            # Scale and predict the cluster
            normalized_data = self.scaler.transform([last_n_days])
            cluster = self.kmeans.predict(normalized_data)[0]

            # Check if this cluster is classified as "UP"
            if cluster in self.up_clusters:
                buy_signals.append(data)
            else:
                # If the cluster is not "UP," sell the position
                self.close(data)

        # Manage position sizing based on the number of buy signals
        if len(buy_signals) > 0:
            weight = 0.9 / len(buy_signals) if len(buy_signals) > 1 else 0.9
            for data in buy_signals:
                self.order_target_percent(data, weight)
        else:
            # No buy signals; close all positions
            for data in self.datas:
                self.close(data)
