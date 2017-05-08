import scipy.io as sio
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics.pairwise import euclidean_distances

data = sio.loadmat('data1.mat')
TraX = data['TrainingX']
TraY = data['TrainingY'].flatten()
TesX = data['TestX']
TesY = data['TestY'].flatten()

dist_tra = euclidean_distances(TraX, TraX)
dist_tes = euclidean_distances(TesX, TraX)

N_tra = TraX.shape[0]
N_tes = TesX.shape[0]

sig2 = dist_tra.sum() / (N_tra**2)
G_tra = np.exp(-dist_tra/sig2)
G_tes = np.exp(-dist_tes/sig2)

w0 = np.zeros(G_tra.shape[0])

Ind_1 = np.random.permutation(5000)[0:2000]
Ind_2 = np.random.permutation(np.arange(5000, 10000))[0:2000]
Ind = np.append(Ind_1, Ind_2)
Ind = np.random.permutation(Ind)
G_BFGS = G_tra[Ind]
G_Y = TraY[Ind]

print('data ready')

def Sigmoid(v):
    return 1/(1+np.exp(-v))

def KernelLogReg(k, w):
    return Sigmoid(k.dot(w))

def J(w, G = G_BFGS, y = G_Y, l=1e-3):
    return -np.log( Sigmoid(y*G.dot(w)) ).sum()/y.shape[0] + l * w.dot(w)

# derivative of J
def d_J(w, G = G_BFGS, y = G_Y, l=1e-3):   
    return 2*l*w - (G.transpose()).dot( (y * (1 - Sigmoid((G.dot(w)*y)) )) ) / y.shape[0]

# accuracy
def accuracy(Y_truth, prob):
    prob[prob >= 0.5] = 1
    prob[prob < 0.5] = -1
    return sum(Y_truth == prob)/len(Y_truth), prob

loss = []
tes_acc = []
j = 0
def step_loss(w):
    global loss, tes_acc, j
    curr_loss = J(w)
    prob_tra = KernelLogReg(G_tra, w)
    prob_tes = KernelLogReg(G_tes, w)
    acc_tra, pred_tra = accuracy(TraY, prob_tra)
    acc_tes, pred_tes = accuracy(TesY, prob_tes)

    loss.append(curr_loss)
    tes_acc.append(acc_tes)

    print('{} \tloss'.format(j) + '\t{}'.format(J(w, G_tra, TraY)) + '\t train accuracy {}'.format(acc_tra) + '\t test accuracy {}'.format(acc_tes))
    j = j + 1

def BFGS(fname, method, eps = 0.005):
    RES = minimize(J, w0,
                   method=method,
                   jac = d_J,
                   options={'disp': True,
                            'maxiter': 1000,
                            'eps':eps},
                   callback=step_loss)

    np.save(method+fname+ '_loss.npy', loss)
    np.save(method+fname+ '_tes_acc.npy', tes_acc)

BFGS(fname = '_h_5', method = 'L-BFGS-B')
BFGS(fname = '_h_5', method = 'BFGS')
