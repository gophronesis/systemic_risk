# Copyright @ 2021 Phronesis. All rights reserved.

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
import pyhsmm
import copy
import pyhsmm.basic.distributions as distributions
import matplotlib.pyplot as plt


class Liquidity:
    """
    
    Estimate systematic liquidity using only close and volume data
    https://www.financialresearch.gov/working-papers/files/OFRwp-2015-11_Systemwide-Commonalities-in-Market-Liquidity.pdf

    Attributes
    ----------
    close : np.ndarray
        daily close data
    volume : np.ndarray
        daily volume data
    window_size : int
        rolling window size
    C : np.ndarray
        market-invariant price impact

    Methods
    -------
    MIPI():
        Calculate Market-invariant Price Impact
    
    hhmm():
        Hierarchical Hidden Markov Model

    """
    
    def __init__(self, close, volume, window_size=20):
        
        """
        
        Initialize Liquidity class
        
        Parameters:
                close (list or np.ndarray): a list of close prices
                volume (list or np.ndarray): a list of volumes
                window_size (int): rolling window size
        
        """
        
        self.close = close
        self.volume = volume
        self.window_size = window_size
        
        if not isinstance(self.close, np.ndarray):
            self.close = np.array(self.close)
            
        if not isinstance(self.volume, np.ndarray):
            self.volume = np.array(self.volume)
        
        if self.close.shape != self.volume.shape:
            print("Close and volume data are not equal in size.")
            return -1
        
        # Check for NaNs
        if np.isnan(self.close).sum() + np.isnan(self.volume).sum() > 0:
            
            print("There are NaNs in the data, by default imputing missing values with sklearn IterativeImputer")
            imp = IterativeImputer(random_state=0)
            imp_df = imp.fit_transform(np.hstack([self.close, self.volume]))
            self.close = imp_df[:, :int(imp_df.shape[1]/2)]
            self.volume = imp_df[:, int(imp_df.shape[1]/2):]
        
        self.close = pd.DataFrame(self.close)
        self.volume = pd.DataFrame(self.volume)
        
    def MIPI(self):
        
        """
        
        Calculate market-invariant price impact
        
        Returns:
            C (list or np.ndarray): market-invariant price impact  
        
        """
    
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.window_size)
        price_roller = self.close.rolling(indexer)
        volume_roller = self.volume.rolling(indexer)

        # Realized volatility
        sigma = price_roller.std()

        # Average realized volume
        V = volume_roller.mean()

        # Impulse size (1% of the volume)
        X = 0.01 * V

        # Speculative activity
        W_i = self.close.to_numpy() * sigma.to_numpy() * V.to_numpy()
        Wi_W = W_i / (0.02 * 40 * 1e6)
        Wi_W_1_3 = np.sign(Wi_W) * (np.abs(Wi_W)) ** (1 / 3)

        # Price impact of impulse size with 1% of volume
        self.C = sigma.to_numpy() / 0.02 * (8.21 / 1e4 * (Wi_W_1_3) + 2.50 / 1e4 * (Wi_W_1_3) * X.to_numpy() / V.to_numpy() / 0.01)
        self.C = self.C[~np.isnan(self.C).any(axis=1)]
        
        # Standardized market-invariant price impact
        self.standardized_C = StandardScaler().fit_transform(self.C)
        
            
    def HHMM(self):
        
        """
        
        Hierarchical Hidden Markov Model
        
        Returns:
            C (list or np.ndarray): market-invariant price impact  
        
        """
        
        obs_dim = self.standardized_C.shape[1]
        Nmax = 3

        obs_hypparams = {'mu_0':np.zeros(obs_dim),
                        'sigma_0':np.eye(obs_dim),
                        'kappa_0':0.3,
                        'nu_0':obs_dim+5}
        dur_hypparams = {'alpha_0':2*30,
                         'beta_0':2}

        obs_distns = [distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
        dur_distns = [distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]

        posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
                alpha=6.,gamma=6., # better to sample over these; see concentration-resampling.py
                init_state_concentration=6., # pretty inconsequential
                obs_distns=obs_distns,
                dur_distns=dur_distns)

        posteriormodel.add_data(self.standardized_C)
        
        self.models = []
        for idx in range(150):
            posteriormodel.resample_model()
            if (idx+1) % 10 == 0:
                print(idx)
                self.models.append(copy.deepcopy(posteriormodel))
                plt.figure()
                posteriormodel.plot()
                plt.gcf().suptitle('HDP-HSMM sampled after %d iterations' % (10*(idx+1)))
                plt.show();
                
        self.states = self.models[-1].stateseqs[0]
        
        avergage_C = np.zeros(Nmax)
        for i in range(Nmax):
            avergage_C[i] = self.standardized_C[self.states == i, :].mean()
            
        self.states = self.states.astype("object")
        for i, idx in enumerate(np.argsort(avergage_C)):
            if idx == 0:
                self.states[self.states == i] = "high"
            elif idx == 1:
                self.states[self.states == i] = "medium"
            else:
                self.states[self.states == i] = "low"
    
    def fit_transform(self):
        self.MIPI()
        self.HHMM()
        return self.states
    
