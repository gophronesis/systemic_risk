import numpy as np
import pandas as pd
from matplotlib import pyplot

class BrownianBridgeSimStoVol():
    
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
    
    def __init__(self, data, N=24, window_size=60, seed = 0, std_threshold=3):
        self.data = data
        self.N = N
        self.window_size = window_size
        self.seed = seed
        self.std_threshold = std_threshold
        np.random.seed(seed)
        
        self.indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window_size)
        self.rolling_std = self.data.rolling(window=self.indexer, 
                                             min_periods=1).std().iloc[:(self.data.shape[0] - 1), :]
        self.rolling_mean = self.data.rolling(window=self.indexer, 
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
    
    def stochastic_volatility(self, repeated_prior_std):
         
        # Find the std of the next moves
        next_move_std = self.data.iloc[self.window_size:, :].to_numpy() - \
                        self.rolling_mean.iloc[:self.rolling_mean.shape[0] - (self.window_size - 1), :].to_numpy()
        next_move_std = np.abs(next_move_std) / self.rolling_std.iloc[:self.rolling_std.shape[0] - (self.window_size - 1), :].to_numpy()

        # Calculate the proportion of days with std above [std_threshold]
        proportion_of_jumps = np.sum(next_move_std > self.std_threshold) / np.prod(self.rolling_std.iloc[:self.rolling_std.shape[0] - (self.window_size - 1), :].shape)

        # Update std on random days based on sampled highly volatile days in the past
        new_std_mask = np.random.uniform(size = repeated_prior_std.shape) < proportion_of_jumps
        repeated_prior_std[new_std_mask] = np.random.choice(next_move_std[next_move_std > 3], size=repeated_prior_std[new_std_mask].shape)
        
        return repeated_prior_std
    
    def simulate(self):
        
        # Simulate a standard brownian motion
        self.simulated_data = self.sample_path_batch(self.M, self.N).T
        
        # Rescale
        prior_std = self.rolling_std.to_numpy().flatten()
        repeated_prior_std = np.tile(prior_std, (self.simulated_data.shape[0], 1))
        repeated_prior_std = self.stochastic_volatility(repeated_prior_std)
        self.simulated_data *= repeated_prior_std
        

        # Shift
        start_price = self.data.iloc[0:-1, :].to_numpy().flatten()
        end_price = self.data.iloc[1:, :].to_numpy().flatten()
        slope_per_step = (end_price - start_price) / self.N # Expected increase every step
        slope_all_steps = np.cumsum(np.ones(self.simulated_data.shape) * slope_per_step, axis=0) # Accumulated increase up to that step
        self.simulated_data += slope_all_steps
        self.simulated_data += start_price
        self.simulated_data = self.simulated_data.reshape(self.N, (self.data.shape[0] - 1), self.data.shape[1])
        self.simulated_data = np.swapaxes(self.simulated_data,0,1)
        self.simulated_data = self.simulated_data.reshape(self.N * (self.data.shape[0] - 1), self.data.shape[1])
        
        return self.simulated_data