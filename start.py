import pandas as pd
import numpy as np
import cvxpy as cp
import scipy
from scipy import optimize
# import nltk
import tqdm
import matplotlib.pyplot as plt
import warnings
import tqdm

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.base import BaseEstimator

import numpy as np
from pykeops.torch import LazyTensor
from pykeops.numpy import LazyTensor as LazyTensor_np

import os
from pathlib import Path


class RBF:
    def __init__(self, sigma=1.):
        self.sigma = sigma  # the variance of the kernel

    def kernel(self, X, Y):
        # Input vectors X and Y of shape Nxd and Mxd
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        X = LazyTensor_np(X[:, None, :])
        Y = LazyTensor_np(Y[None, :, :])  # (1, M, d)

        # Compute the squared distance matrix
        D_ij = ((X - Y) ** 2).sum(axis=2)  # (N, M)

        # Compute the RBF kernel
        K_ij = (-D_ij / (2 * self.sigma ** 2)).exp()

        return K_ij

    def __str__(self):
        return f'RBF {self.sigma}'


class RBF_native:
    def __init__(self, sigma=1.):
        self.sigma = sigma  ## the variance of the kernel
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        return np.exp(-np.sum((X[:,None]-Y[None])**2, axis=-1)/2/self.sigma**2)
    def __str__(self):
        return f'RBF {self.sigma}'

class Linear:
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        return np.dot(X,Y.T)
    def __str__(self):
        return 'Linear'

class Jaccard:
    def __init__(self, renorm=False):
        self.renorm = renorm
    def kernel(self,X,Y):
        if self.renorm:
            X[X>0] = 1
            Y[Y>0] = 1
        ## Input vectors X and Y of shape Nxd and Mxd
        return 2*np.dot(X,Y.T)/(np.linalg.norm(X,axis=-1)[:,None]+np.linalg.norm(Y,axis=-1)[None])
    def __str__(self):
        return f'Jaccard renorm={self.renorm}'


class KernelSVC(BaseEstimator):

    def __init__(self, C, kernel, epsilon=1e-3):
        self.type = 'non-linear'
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.support = None  # support vectors
        self.epsilon = epsilon
        self.norm_f = None

    def fit(self, X, y):
        N = len(y)
        self.K = self.kernel(X, X)
        yKy = y[:, None] @ y[None, :] * self.K

        # Define the variable alpha
        alpha = cp.Variable(N)


        # Define the objective function
        objective = cp.Minimize(0.5 * cp.quad_form(alpha, cp.Parameter(shape=yKy.shape, value=yKy, PSD=True)) - cp.sum(alpha))

        # Define the constraints
        constraints = [
            cp.sum(cp.multiply(alpha, y)) == 0,  # Equality constraint
            alpha >= 0,              # Non-negativity constraint
            alpha <= self.C          # Box constraint
        ]

        # Define the problem
        problem = cp.Problem(objective, constraints)

        # Solve the problem
        problem.solve()

        # Extract the solution
        self.alpha = alpha.value

        # Assign the required attributes
        conditions = (self.alpha > self.epsilon)
        self.parameters = self.alpha[conditions] * y[conditions]
        self.support = X[conditions]
        support_vector_indices = np.where(conditions)[0]
        self.b = np.mean(y[support_vector_indices] -
                         np.dot(self.K[support_vector_indices][:, support_vector_indices], self.alpha[support_vector_indices] * y[support_vector_indices]))

        self.norm_f = (self.alpha * y).dot(self.K).dot(self.alpha * y)

    ### Implementation of the separating function $f$
    def separating_function(self, x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        # Kernel
        K = self.kernel(x, self.support)
        return K.dot(self.parameters)

    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return 2 * (d + self.b > 0) - 1

def get_bags_overlapped(df, size_bag, col_name="seq"):
    return df[col_name].apply(lambda x: [x[i:i+size_bag] for i in range(len(x) - size_bag + 1)])

def get_bags(df, size_bag, col_name='seq'):
    return df[col_name].apply(lambda x: [x[i:i+size_bag] for i in range(0, len(x), size_bag)])

def get_all_posibilities(size_bag):
    tokens = ["A", "C", "G", "T"]
    all_subsequence = tokens.copy()
    for i in range(size_bag - 1):
        all_subsequence = [a + b for a in all_subsequence for b in tokens]
    return all_subsequence


data_directory = Path('data/')
nb_tr_to_fit = 3

dfs = [data_directory / f'Xtr{i}.csv' for i in range(nb_tr_to_fit)]
dfs = [pd.read_csv(df) for df in dfs]
df_y = [data_directory / f'Ytr{i}.csv' for i in range(nb_tr_to_fit)]
df_y = [pd.read_csv(df) for df in df_y]

for i in range(nb_tr_to_fit):
    dfs[i]["Bound"] = df_y[i]["Bound"]
    dfs[i]["y"] = 2*dfs[i]["Bound"] - 1

df_test = [data_directory / f'Xte{i}.csv' for i in range(nb_tr_to_fit)]
df_test = [pd.read_csv(df) for df in df_test]


size_bag = 5
print(f"Size of the latent representation: {4**size_bag}")
# df['bags'] = df['seq'].apply(lambda x: [x[i:i+size_bag] for i in range(0, len(x), size_bag)])
for df in dfs:
    df['bags'] = get_bags_overlapped(df, size_bag)
    all_posibilities = get_all_posibilities(size_bag)
    df['counts'] = df['bags'].apply(lambda x: [x.count(subseq) for subseq in all_posibilities])

for df in df_test:
    df['bags'] = get_bags_overlapped(df, size_bag)
    all_posibilities = get_all_posibilities(size_bag)
    df['counts'] = df['bags'].apply(lambda x: [x.count(subseq) for subseq in all_posibilities])

models = [KernelSVC(C=0.005, kernel=Linear().kernel) for _ in range(nb_tr_to_fit)]
for i in range(nb_tr_to_fit):
    models[i].fit(np.stack(dfs[i]['counts'].values), np.array(dfs[i]['y']))

for i in range(nb_tr_to_fit):
    y_pred = models[i].predict(np.stack(df_test[i]['counts'].values))
    df_test[i]["Bound"] = (y_pred + 1) / 2

# concat
df_test_final = pd.concat(df_test)
df_test_final["Bound"] = df_test_final["Bound"].astype(int)
df_test_final[["Id", "Bound"]].to_csv("predictions/Yte_svm_mismatchkernel.csv", index=False)