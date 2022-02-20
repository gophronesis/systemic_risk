import numpy as np
import pandas as pd
from matplotlib import pyplot

class BrownianBridgeSim():
    
    """
    Simulating multivariate time-series between two available time points using brownian bridges.
    The mean and std of the random walk consists of:
        1. Linear slope between two existing time points
        2. Estimated/observed std of a sliding window
    
    Input:
        data (np.ndarray): dim [Time x Variable/Features]
            * Multivariate time series data
        
        N (int): scalar, default 100
            * Number of data points to simulate between two existing data points
            
        window_size (int): scalar, default 5
            * window size used to estimate std
            
        seed (int): scalar, default 0
            * random seed
    """
    
    def __init__(self, data, N=100, window_size=5, seed = 0):
        self.data = data
        self.N = N
        self.window_size = window_size
        self.seed = seed
        np.random.seed(seed)
        
        self.indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=5)
        self.rolling_std = self.data.rolling(window=self.indexer, 
                                             min_periods=1).std().iloc[:(self.data.shape[0] - 1), :]
        self.M = (self.data.shape[0] - 1) * self.data.shape[1]

        
    def sample_path_batch(self, M, N):
        dt = 1.0 / (N -1)
        dt_sqrt = np.sqrt(dt)
        B = np.empty((M, N), dtype=np.float32)
        B[:, 0] = 0
        for n in range(N - 2): 
            t = n * dt
            xi = np.random.randn(M) * dt_sqrt
            B[:, n + 1] = B[:, n] * (1 - dt / (1 - t)) + xi
        B[:, -1] = 0
        return B
    
    def simulate(self):
        
        # Simulate a standard brownian motion
        self.simulated_data = self.sample_path_batch(M, N).T
        
        # Rescale
        prior_std = self.rolling_std.to_numpy().flatten()
        repeated_prior_std = np.tile(prior_std, (self.simulated_data.shape[0], 1))
        self.simulated_data *= repeated_prior_std

        # Shift
        start_price = self.data.iloc[0:-1, :].to_numpy().flatten()
        end_price = self.data.iloc[1:, :].to_numpy().flatten()
        slope_per_step = (end_price - start_price) / N # Expected increase every step
        slope_all_steps = np.cumsum(np.ones(self.simulated_data.shape) * slope_per_step, axis=0) # Accumulated increase up to that step
        self.simulated_data += slope_all_steps
        self.simulated_data += start_price
        self.simulated_data = self.simulated_data.reshape(N, (self.data.shape[0] - 1), self.data.shape[1])
        self.simulated_data = np.swapaxes(self.simulated_data,0,1)
        self.simulated_data = self.simulated_data.reshape(N * (self.data.shape[0] - 1), self.data.shape[1])
        return self.simulated_data