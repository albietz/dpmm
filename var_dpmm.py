import numpy as np
import scipy.sparse as sp
from scipy.special import psi
import sys

ap_data = '../ap/ap.dat'
ap_vocab = '../ap/vocab.txt'

def load_data(ap_data, ap_vocab):
    n = len(open(ap_data).readlines())
    m = len(open(ap_vocab).readlines())

    X = sp.lil_matrix((n,m))

    for i, line in enumerate(open(ap_data)):
        words = line.split()
        idxs = []
        vals = []
        for w in words[1:]:
            idx, val = map(int, w.split(':'))
            idxs.append(idx)
            vals.append(val)
        
        X[i,idxs] = vals

    return X

def logsumexp(a, axis=None):
    a_max = np.max(a, axis=axis)
    try:
        return a_max + np.log(np.sum(np.exp(a - a_max), axis=axis))
    except:
        return a_max + np.log(np.sum(np.exp(a - a_max[:,np.newaxis]), axis=axis))

def var_dpmm_multinomial(X, alpha, base_dirichlet, T=50, n_iter=100):
    N, M = X.shape

    # variational multinomial parameters for z_n
    phi = 1./T * np.matrix(np.ones((T,N)))

    # variational beta parameters for V_t
    gamma1 = np.matrix(np.zeros((T-1,1)))
    gamma2 = np.matrix(np.zeros((T-1,1)))

    # variational dirichlet parameters for \eta_t
    tau = np.matrix(np.zeros((T,M)))

    for it in range(n_iter):
        print it
        gamma1 = 1. + np.sum(phi[:T-1,:], axis=1)
        phi_cum = np.cumsum(phi[:0:-1,:], axis=0)[::-1,:]
        gamma2 = alpha + np.sum(phi_cum, axis=1)

        tau = base_dirichlet + phi * X

        lV1 = psi(gamma1) - psi(gamma1 + gamma2)  # E_q[log V_t]
        lV1 = np.vstack((lV1, 0.))
        lV2 = psi(gamma2) - psi(gamma1 + gamma2)  # E_q[log (1-V_t)]
        lV2 = np.cumsum(np.vstack((0., lV2)), axis=0)  # \sum_{i=1}^{t-1} E_q[log (1-V_i)]

        eta = psi(tau) - psi(np.sum(tau, axis=1)) # E_q[eta_t]

        S = lV1 + lV2 + eta * X.T
        S = S - logsumexp(S, axis=0)
        phi = np.exp(S)

        # print_top_words_for_topics(top_topics_of_document(0, phi, n_topics=5),tau)

    return gamma1, gamma2, tau, phi

def print_top_words_for_topics(topics, tau, n_words=10):
    voc = np.array(open(ap_vocab).read().strip().split('\n'))

    if isinstance(topics, tuple):
        for topic, prob in zip(*topics):
            idx = np.argsort(tau[topic,:].A1)[::-1]
            print '{} ({}): {}'.format(topic, float(prob), ', '.join(voc[idx[:n_words]]))
    else:
        for topic in topics:
            idx = np.argsort(tau[topic,:].A1)[::-1]
            print '{}: {}'.format(topic, ', '.join(voc[idx[:n_words]]))

def top_topics_of_document(n, phi, n_topics=None):
    idx = np.argsort(phi[:,n].A1)[::-1]
    return idx[:n_topics], phi[idx[:n_topics],n]

if __name__ == '__main__':
    X = load_data(ap_data, ap_vocab)
    N, M = X.shape

    alpha = 1
    base_dirichlet = 0.1 * np.ones(M)

    g1, g2, tau, phi = var_dpmm.var_dpmm_multinomial(X, alpha, base_dirichlet, T=20, n_iter=50)
