#!/usr/bin/env python
# coding: utf-8

# In[1]:


import import_ipynb
import general
import Bestkernel
import numpy as np
from itertools import product
from tqdm import tqdm
import pickle as pkl
import os
import pandas as pd
from SVM import C_SVM


build_kernel = False  # Build a kernel
check_method = False  # Use a particular method
check_NLCK   = False  # Use NLCK algorithm
check_CVNLCK = False  # Use cross validation on NLCK hyperparameters
check_other  = False  # Free

if __name__ == '__main__':
    if build_kernel:
        methods = ['gappy_kernel_k3_g1', 'mismatch_lernel_k5_m1', 'weighted_degree_d10']
        for method in methods:
            X_train, y_train, X_val, y_val, X_test, K, ID = general.data_bring(method=method, replace=True)
            # Put replace = False not to erase the previous saves

    elif check_method:
        method = 'mismatch_lernel_k6_m1'
        algo = 'Kernelridge'
        solver = 'CVX'
        data, data1, data2, data3, K, ID = general.data_bring([method])
        Cs = np.sort([i * 10 ** j for (i, j) in product(range(1, 10), range(-2, 1))])
        # Perform cross validation on data set 1 (TF = 1)
        general.cross_validation(Ps=Cs, data=data1, algo=algo, solver=solver, kfolds=3, K=K, ID=ID)



    elif check_NLCK:
        methods = ['spectrum_kernel_k6', 'spectrum_kernel_k5', 'spectrum_kernel_k4']
        data, data1, data2, data3, kernels, ID = general.data_bring(methods)
        X_train_1, y_train_1, X_val_1, y_val_1, X_test_1, kernels_1, ID_1 = general.rechange(data1, kernels, ID)
        Km1 = Bestkernel.nonlinear_kernel(X_train_1, y_train_1, ID_1, kernels_1, C=1e-2, eps=1e-9, degree=2).get_K()
        Cs = np.sort([i * 10 ** j for (i, j) in product(range(1, 10), range(-3, 5))])
        general.cross_validation(Ps=Cs, data=data1, algo='Kernelridge', kfolds=5, K=Km1, ID=ID_1)

    elif check_CVNLCK:
        methods = ['mismatch_lernel_k3_m1', 'weighted_degree_k5', 'string_kernel_l1_k3']
        Cs_NLK = [1e-3, 1e-2, 0.1, 1, 10, 100]
        Cs_KLR = np.concatenate((np.linspace(0.01, 0.1, 19), np.linspace(0.1, 1, 91), np.linspace(1, 10, 19)))
        degrees = [1]
        lbdas = [1, 5, 10, 50, 100]
        Bestkernel.cross_validation(k=1, methods=methods, Cs_NLK=Cs_NLK, Cs_KLR=Cs_KLR, degrees=degrees, lambdas=lbdas)

    elif check_other:
        pass


# In[ ]:




