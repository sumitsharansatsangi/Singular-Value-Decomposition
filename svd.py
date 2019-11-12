# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
#import scipy as sp
from sklearn.metrics import mean_squared_error
#A=np.matrix([[11,24,32],[12,4,6],[3,7,2],[5,4,9]])
A=pd.read_csv("/home/sumit/sumit/downlod 1/train.csv")
B=A.dot(A.T)
sig_vec_u=np.linalg.eig(B)
idx_u=sig_vec_u[0].argsort()[-1::-1]
sig_u=sig_vec_u[0][idx_u]
sigma_u=np.diag(sig_u)
sigma_u=np.sqrt(sigma_u)
u=sig_vec_u[1][idx_u]
#A=np.matrix([[11,24,32],[12,4,6],[3,7,2],[5,4,9]])
C=A.T.dot(A)
sig_vec_v=np.linalg.eig(C)
idx_v=sig_vec_v[0].argsort()[-1::-1]
sig_v=sig_vec_v[0][idx_v]
sigma_v=np.diag(sig_v)
sigma_v=np.sqrt(sigma_v)
v=sig_vec_v[1][idx_v]
col=min(u.shape[0],v.shape[0])
sigma=sigma_u[:,0:col]
u=np.linalg.qr(u)
v=np.linalg.qr(v)
A_p=u[0].dot(sigma).dot(v[0].T)
err=np.linalg.norm(A-A_p)
err=err/A.shape[0]
print(err)
#print(mean_squared_error(A,A_p))