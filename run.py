#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import import_ipynb
from Bestkernel import nonlinear_kernel
from general import rechange, preict_send_to_output, data_bring
from SVM import C_SVM

# We got the best result with this 9 kernels:
methods = ['spectrum_kernel_k4', 'spectrum_kernel_k5', 'spectrum_kernel_k6', 'mismatch_lernel_k4_m1', 'mismatch_lernel_k5_m1', 'mismatch_lernel_k6_m1', 'weighted_degree_d4', 'weighted_degree_d5', 'weighted_degree_d10']
#methods = ['SP_k4']

# Import data
data, data1, data2, data3, kernels, ID = data_bring(methods)

# Use the algorithm on the first data set with the corresponding hyperparameters
print('\n\n')
X_train_1, y_train_1, X_val_1, y_val_1, X_test_1, kernels_1, ID_1 = rechange(data1, kernels, ID)
Km1 = nonlinear_kernel(X_train_1, y_train_1, ID_1, kernels_1, C=1, eps=1e-9, degree=3).get_K(fnorm=5, n_iter=50)
svm1 = C_SVM(Km1, ID_1, C=1.9, solver='CVX')
svm1.fit(X_train_1, y_train_1)

# Use the algorithm on the second data set with the corresponding hyperparameters (see the report, table 1)
print('\n\n')
X_train_2, y_train_2, X_val_2, y_val_2, X_test_2, kernels_2, ID_2 = rechange(data2, kernels, ID)
Km2 = nonlinear_kernel(X_train_2, y_train_2, ID_2, kernels_2, C=10, eps=1e-9, degree=4).get_K(fnorm=5, n_iter=50)
svm2 = C_SVM(Km2, ID_2, C=2.1, solver='CVX')
svm2.fit(X_train_2, y_train_2)

# Use the algorithm on the third data set with the corresponding hyperparameters (see the report, table 1)
print('\n\n')
X_train_3, y_train_3, X_val_3, y_val_3, X_test_3, kernels_3, ID_3 = rechange(data3, kernels, ID)
Km3 = nonlinear_kernel(X_train_3, y_train_3, ID_3, kernels_3, C=1e-2, eps=1e-9, degree=3).get_K(fnorm=1, n_iter=50)
svm3 = C_SVM(Km3, ID_3, C=3, solver='CVX')
svm3.fit(X_train_3, y_train_3)

# See scores on validation set
print('\n\nAccuracy on validation set 1: {:0.4f}'.format(svm1.score(svm1.predict(X_val_1), y_val_1)))
print('Accuracy on validation set 2: {:0.4f}'.format(svm2.score(svm2.predict(X_val_2), y_val_2)))
print('Accuracy on validation set 3: {:0.4f}'.format(svm3.score(svm3.predict(X_val_3), y_val_3)))

# Compute predictions
y_pred = preict_send_to_output([svm1, svm2, svm3], [X_test_1, X_test_2, X_test_3])
print('\n\nPredictions ok')


# In[ ]:




