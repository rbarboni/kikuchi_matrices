import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.special as ss
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from joblib import Parallel, delayed

def subsets(size, n):
    res = []
    assert size >= 0
    assert size <= n
    if size == 0:
        return [np.zeros(n)]
    elif size == n:
        return [np.ones(n)]
    else:
        sets = subsets(size, n-1)
        for set in sets:
            res.append(np.append(set,0))
        sets = subsets(size-1, n-1)
        for set in sets:
            res.append(np.append(set,1))
        return res

def random_matrix(n, l, p):
    set_indices = subsets(l, n)
    num_indices = len(set_indices)

    G = np.random.randn(*(n for _ in range(p)))
    data = []
    i_index = []
    j_index = []

    for i in range(num_indices):
        S = set_indices[i]
        for j in range(i + 1, num_indices):
            T = set_indices[j]
            E = (S + T) % 2
            E_ind = np.where(E > 0.5)[0]
            if len(E_ind) == p:
                g = G[tuple(E_ind)]
                data.append(g)
                i_index.append(i)
                j_index.append(j)
                data.append(g)
                i_index.append(j)
                j_index.append(i)

    return sparse.coo_matrix((data, (i_index, j_index)))

def sigma2(n, l, p):
    return ss.binom(l, p//2) * ss.binom(n-l, p//2)

def eigenvalue_distribution(n, l, p):
    Z = random_matrix(n, l, p)
    N = ss.binom(n, l)
    eigs, _ = linalg.eigsh(Z, k=N)
    plt.hist(eigs, bins=int(N**(0.6)), density=True)
    plt.show()

def norm(n, l, p):
    Z = random_matrix(n, l, p)
    sigma = sigma2(n, l, p)**0.5
    eigs = linalg.svds(Z, k=1, which='LM', tol=0.2*sigma, return_singular_vectors=False)
    return abs(eigs[0])

def norm_evolution(n_list, l, p, sample_size=-1):
    print('preparing norm evolution plot')
    res = np.zeros(len(n_list))
    if sample_size < 0:
        n_jobs = os.cpu_count()
    else:
        n_jobs = sample_size
    for i, n in tqdm(enumerate(n_list)):
        print(f'\n n = {n}, N = {ss.binom(n,l)}')
        norms = Parallel(n_jobs=n_jobs)(delayed(norm)(n, l, p) for _ in range(n_jobs))
        res[i] = np.array(norms).mean() / sigma2(n, l, p)**0.5
    plt.plot(n_list, res, '+')
    plt.show()

def regression_plot(n_list, l_list, p):
    res = []
    N_list = []
    for l in l_list:
        for n in n_list:
            N = ss.binom(n, l)
            N_list.append(N)
            for _ in range(2):
                mean_norm = 0
                Z = random_matrix(n, l, p)
                eigs, _ = linalg.eigsh(Z, k=1, which='LM')
                mean_norm += eigs[0] / sigma2(n, l, p)**0.5
            res.append(0.5*mean_norm)
    idx = np.argsort(N_list)
    return np.array(N_list)[idx], np.array(res)[idx]

def pairing_moment(n, l, p, pairing):
    sigma = sigma2(n, l, p)**0.5
    N = int(ss.binom(n, l))
    matrix_list = [random_matrix(n, l, p).tocsr() / sigma for _ in range(len(pairing))]
    ordered_matrix_list = [sparse.csr_matrix((N, N)) for _ in range(2 * len(matrix_list))]
    for i, pair in enumerate(pairing):
        ordered_matrix_list[pair[0]] += matrix_list[i]
        ordered_matrix_list[pair[1]] += matrix_list[i]
    matrix_product = sparse.eye(N, format='csr')
    for matrix in ordered_matrix_list:
        matrix_product = matrix_product @ matrix
    return matrix_product.toarray().trace() / N

def enumerate_pairings(set):
    if len(set) == 2:
        return [[(set[0], set[1])]]
    elif len(set) > 2:
        res = []
        for i in range(1, len(set)):
            pairing_list = enumerate_pairings(set[1:i]+set[i+1:])
            for pairing in pairing_list:
                pairing.append((set[0], set[i]))
            res = pairing_list + res
        return res

def cross(pair0, pair1):
    return ((pair0[0] - pair1[0]) * (pair0[1] - pair1[1]) > 0) and ((pair0[1] - pair1[0]) * (pair0[0] - pair1[1]) < 0)

def k_crossings(k, pairing):
    if len(pairing) < k:
        return []
    if k == 1:
        return [[pair] for pair in pairing]
    res = []
    pair0 = pairing[0]
    crossings_list = k_crossings(k-1, pairing[1:])
    for crossing in crossings_list:
        cond = True
        for pair in crossing:
            if not cross(pair0, pair):
                cond = False
        if cond:
            res.append([pair0]+crossing)
    for crossing in k_crossings(k, pairing[1:]):
        res.append(crossing)
    return res






