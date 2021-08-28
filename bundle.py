import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import cvxpy as cp
from tqdm import tqdm
import ray

class DataManager():

    def __init__(self, data_dir = './Data'):
        self.data_dir = data_dir
        self.ratio_returns = pd.read_pickle(data_dir + "/ratio_returns.pkl")
        self.dollar_volume = pd.read_pickle(data_dir + "/dollar_volume.pkl")
        self.ric_symbols = pd.read_pickle(data_dir +  "/ric_symbols.pkl")
        self.current_date_idx = None
        self.universe = []

    def set_current_idx(self, num=2730):
        self.current_date_idx = num

    def create_train_data(self,
        window_size = 252,
        std_threshold = 1e-4,
        clip_lower_value = -0.2,
        clip_upper_value = 0.2
        ):    
        if self.current_date_idx < window_size:
            print(f"Not enough training data, please use a later date!")
            
        history_window_idx = list(range(self.current_date_idx - window_size, self.current_date_idx))
            
        current_data = self.ratio_returns.iloc[history_window_idx, :]    
        active_set = np.where(current_data.notna().all())[0]
        
        tmp = current_data.iloc[:, active_set]
        tmp_clipped = tmp.clip(clip_lower_value, clip_upper_value, axis=1)
        active_set_new = active_set[np.where(tmp_clipped.std(axis=0) > std_threshold)[0]]
        
        # Create training dataset     
        train_data = self.ratio_returns.iloc[history_window_idx, active_set_new]
        train_data_clipped = train_data.clip(clip_lower_value, clip_upper_value, axis=1)
        
        # Create traing volume dataset
        train_volume = self.dollar_volume.iloc[history_window_idx, active_set_new]
        train_volume_fillna = train_volume.fillna(train_volume.median())
        train_volume_clipped = train_volume_fillna.clip(1e2, 1e8, axis=1)
        self.universe = active_set_new
            
        return train_data_clipped.to_numpy(), train_volume_clipped.to_numpy()


    def create_test_data(self, test_horizon = 21):           
        if self.current_date_idx + test_horizon > self.ratio_returns.shape[0]:
            print(f"Not enough testing data, please use a shorter date!")    
        
        test_window_idx = list(range(self.current_date_idx + 1, self.current_date_idx + test_horizon + 1))
        test_data = self.ratio_returns.iloc[test_window_idx, self.universe]   
        return test_data.to_numpy()



class BundleConstructor():

    def __init__(self, data_matrix, volume_matrix):
        X = data_matrix
        self.volume_matrix = volume_matrix
        self.T, self.N = X.shape
        X_std = np.sqrt(np.std(X, axis=0, keepdims=True) ** 2 + np.mean(X, axis=0, keepdims=True) ** 2)
        X_standardized = X / X_std
        self.R = X_standardized.T @ X_standardized / self.T
        self.eigvals, self.eigvecs = np.linalg.eigh(self.R)  
        xi_clipped = np.where(self.eigvals >= (1 + np.sqrt(self.N / float(self.T)))**2, self.eigvals, np.nan)
        gamma = float(self.R.trace() - np.nansum(xi_clipped))
        gamma /= np.isnan(xi_clipped).sum()
        self.xi_clipped = np.where(np.isnan(xi_clipped), gamma, xi_clipped)
        self.median_volume = np.median(volume_matrix, axis=0)
        


    def correlation_matrix(self, sqrt=True):
        
        if sqrt:
            R_clipped = self.eigvecs @ np.diag(np.sqrt(self.xi_clipped)) @ self.eigvecs.T
            R_inv = self.eigvecs @  np.diag(1 / np.sqrt(self.xi_clipped)) @ self.eigvecs.T
        else:
            R_clipped = self.eigvecs @ np.diag(self.xi_clipped) @ self.eigvecs.T 
            tmp = 1./np.sqrt(np.diag(R_clipped))
            R_clipped *= tmp
            R_clipped *= tmp.reshape(-1, 1)
            R_inv = self.eigvecs @  np.diag(1 / self.xi_clipped) @ self.eigvecs.T

        return R_clipped, R_inv


    def volume_graph_filter(self, alpha=0.5):
        adjacency_matrix = np.full([self.N, self.N], 1)
        for j in range(self.N):
            adjacency_matrix[np.where(self.median_volume < alpha * self.median_volume[j]), j] = 0
        return adjacency_matrix * adjacency_matrix.T
 

    def thresholding_prior(self, E_volume, alpha=1, beta=1):
        Lam = abs(self.R)**alpha
        Lam[np.where(E_volume == 0)] = beta
        return Lam - np.diag(np.diag(Lam))

    def soft_threshold_search(self, 
        S, 
        Lam, 
        rhos_pool=np.arange(0.0001, 1, 0.0001), 
        target_sparsity=0.02
        ):

        left, right = 0, len(rhos_pool) - 1
        ascending = True
            
        def sparsity_level(S):
            return (np.count_nonzero(S) - S.shape[0]) / (S.shape[0] * (S.shape[0] - 1))

        def soft_threshold(S, Lam, rho):
            return np.maximum(np.abs(S) - rho * Lam, np.zeros_like(S)) * np.sign(S)

        def evaluate(idx):
            return sparsity_level(soft_threshold(S, Lam, rhos_pool[idx]))
        
        if evaluate(left) > evaluate(right):
            ascending = False
        
        while left < right - 1:
            mid = left + (right - left) // 2        
            if evaluate(mid) == target_sparsity:
                index = mid
            if evaluate(mid) < target_sparsity:
                if ascending:
                    left = mid
                else:
                    right = mid
            if evaluate(mid) > target_sparsity:
                if ascending:
                    right = mid
                else:
                    left = mid        
        if evaluate(left) >= target_sparsity:
            index = left
        if evaluate(right) >= target_sparsity:
            index = right
        return np.sign(soft_threshold(S, Lam, rhos_pool[index]))



    def graph_search(self, S, E, L_in, L_out):
        
        volume_idxs = np.argsort(-1 * self.median_volume)
        
        E = E[np.ix_(volume_idxs, volume_idxs)]
        S = S[np.ix_(volume_idxs, volume_idxs)]   
        N = E.shape[0]
        np.fill_diagonal(E, 0)
        degree_in = np.zeros(N)
        E_out = np.eye(N)
            
        for j in range(N):
            Cj = np.where(np.logical_and(E[:, j] == 1, degree_in < L_in))[0]
            
            if len(Cj) < L_out[j]:
                Aj = Cj
            else:            
                Aj = Cj[np.argsort(-1 * abs(S[Cj, j]))[:L_out[j]]]
            
            E_out[Aj, j] = 1
            degree_in[Aj] += 1
            
        inv_idxs = np.argsort(volume_idxs)
        E_bundle = E_out[np.ix_(inv_idxs, inv_idxs)]
        return E_bundle


    def sparse_inverse(self, S, adjacency_matrix, lamb=0.1):
        S = (S + S.T) / 2
        N = S.shape[0]
        B = np.zeros_like(S) 
        for j in range(N):        
            
            idxs_j = np.nonzero(adjacency_matrix[:,j])[0]
            Sj_sub = S[idxs_j,:][:,idxs_j]
            ej = np.zeros((N,1))
            ej[j] = 1
            ej_sub = ej[idxs_j]
            
            M = Sj_sub.shape[0]  
            A = np.ones((1, M))
            b = 0
            # optimization
            x = cp.Variable((M, 1))
            cons = [A @ x == b]
            obj = cp.Minimize((1 / 2) * cp.quad_form(x, Sj_sub) - \
                            ej_sub.T @ x  + lamb * cp.sum(cp.abs(x)))
            prob = cp.Problem(obj, cons)
            prob.solve()
            B[idxs_j, j] = np.array(x.value).reshape(-1,)

        diagonal_elements = np.diag(B)
        B_normalized = B @ np.diag(1/diagonal_elements)
            
        return B, B_normalized


class Evaluator():

    def __init__(self):
        self.dm = DataManager()
        print("init a new Data Manager here")
    
    def create_bundles(self, start_idx=2730, end_idx=2980):
        bundle_list = []
        Z_list = []
        for t in range(start_idx, end_idx):
            self.dm.set_current_idx(t)
            X_train, volume_train = self.dm.create_train_data()
            X_test = self.dm.create_test_data()
            X_filled = np.nan_to_num(X_test, 0)
            bc = BundleConstructor(X_train, volume_train)
            R_sqrt, R_sqrt_inv = bc.correlation_matrix()
            E_volume = bc.volume_graph_filter()
            Lam = bc.thresholding_prior(E_volume)
            E_threshold = bc.soft_threshold_search(R_sqrt, Lam)
            E_bundle = bc.graph_search(R_sqrt_inv, E_threshold, L_in= bc.N * [40], L_out= bc.N *[20])
            B, _ = bc.sparse_inverse(R_sqrt, E_bundle)
            bundle_list.append(B) 
            Z_list.append(X_filled @ B)
        return bundle_list, Z_list

