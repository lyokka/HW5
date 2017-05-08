import numpy as np

# Sigmoid function
def Sigmoid(v):
    return 1/(1+np.exp(-v))

# Kernel regression
def KernelLogReg(k, w):
    return Sigmoid(k.dot(w))

# Loss J
def J(w, G, y, l):
    return -np.log( Sigmoid(y*G.dot(w)) ).sum()/y.shape[0] + l * w.dot(w)
#    return -np.log( Sigmoid(y*G.dot(w)) ).sum() + l * w.dot(w)


# derivative of J
def d_J(w, G, y, l):   
    return 2*l*w - (G.transpose()).dot( (y * (1 - Sigmoid((G.dot(w)*y)) )) ) / y.shape[0]
#    return 2*l*w - (G.transpose()).dot( (y * (1 - Sigmoid((G.dot(w)*y)) )) )


# accuracy
def accuracy(Y_truth, prob):
    prob[prob >= 0.5] = 1
    prob[prob < 0.5] = -1
    return sum(Y_truth == prob)/len(Y_truth), prob

def SGD(w, G_tra, G_tes, TraY, TesY, fname, p, eps=0, h=0.005, l=1e-3, niter=10000):    

    loss = []
    tes_acc = []
    j = 0 # counter
    
    while True:
        ind = np.random.permutation(G_tra.shape[0])[:p]
        G = G_tra[ind, :]
        Y = TraY[ind]
        d = d_J(w, G, Y, l)
        w = w - h * d
        loss.append(J(w, G_tra, TraY, l))

        prob_tra = KernelLogReg(G_tra, w)
        prob_tes = KernelLogReg(G_tes, w)
        acc_tra, pred_tra = accuracy(TraY, prob_tra)
        acc_tes, pred_tes = accuracy(TesY, prob_tes)

        tes_acc.append(acc_tes)

        if j % 1 == 0:
            print('{} \tloss'.format(j) + '\t{}'.format(J(w, G_tra, TraY, l)) + '\t train accuracy {}'.format(acc_tra) + '\t test accuracy {}'.format(acc_tes))

        j += 1
        #print('d_J norm {}'.format(np.linalg.norm(d)))

        if (np.linalg.norm(d) <= eps) | (j>niter):
            print('stop {}'.format(j))
            break

    np.save(fname+'_loss.npy', loss)
    np.save(fname+'_tes_acc.npy', tes_acc)
