from p1 import *
import scipy.io as sio
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import warnings

# compress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# load data, sigma^2, and kernel
data = sio.loadmat('data1.mat')

# prepare train, test data and label
TraX = data['TrainingX']
TraY = data['TrainingY'].flatten()
TesX = data['TestX']
TesY = data['TestY'].flatten()

dist_tra = euclidean_distances(TraX, TraX)
dist_tes = euclidean_distances(TesX, TraX)

N_tra = TraX.shape[0]
N_tes = TesX.shape[0]

sig2 = dist_tra.sum()/(N_tra**2)
G_tra = np.exp(-dist_tra/sig2)
G_tes = np.exp(-dist_tes/sig2)

# Gradient descent(constant step)
w0 = np.zeros(G_tra.shape[0])

# GD
#SGD(w0, G_tra, G_tes, TraY, TesY, fname = 'GD_h_5', p = N_tra, eps = 1e-2, h = 0.005, niter=20000)
#print('\n\n\nGD end\n\n\n')

# SGD with batchsize 100
#SGD(w0, G_tra, G_tes, TraY, TesY, fname = 'SGD_p100_h_5', p = 100, eps = 1e-2, h = 0.005, niter=20000)
#print('\n\n\nSGD with batchsize 100 end\n\n\n')

# SGD with batchsize 1
SGD(w0, G_tra, G_tes, TraY, TesY, fname = 'SGD_p1_h_5_70000', p = 1, eps = 0, h= 1e-2, niter=70000)
#print('\n\n\nSGD with batchsize 1 end\n\n\n')
