import numpy as np
import matplotlib.pyplot as plt
import scipy
import time

from sklearn import datasets
from scipy.stats import entropy
from itertools import combinations
from math import isnan


def pairwise_prob(x, perplexity):
    dist = scipy.spatial.distance.pdist(x)
    n = np.size(x, 0)
    pair_ls = [pair for pair in combinations(range(n), 2)]

    p = np.zeros((n, n))
    start_time = time.time()
    for i in range(n):

        p[i] = conditional_prob(i, n, dist, perplexity)

    print('compute p given i: {} seconds'.format(time.time() - start_time))
    return p


def conditional_prob(i, n, dist, perplexity):
    sigma = 0.1
    tolerance = 0.01

    sigma_low = 0
    sigma_high = np.NaN
    while True:
        p = p_given_i(i, n, dist, sigma)

        error = 2**entropy(p) - perplexity
        if abs(error) > tolerance:
            if error > 0:
                sigma_high = sigma
                sigma = (sigma_low + sigma)/2
            else:
                sigma_low = sigma
                sigma = sigma*2 if isnan(sigma_high) else (sigma_high + sigma)/2
        else:
            break

    return p


# compute p_ji given i for all j
def p_given_i(i, n , dist, sigma):

    p = np.zeros(n)
    pair_ls = [pair for pair in combinations(range(n), 2)]
    k = 2*sigma**2
    exp_sum = 0

    dist_ls = []

    for pair_index in range(len(pair_ls)):
        if i in pair_ls[pair_index]:
            dist_ls.append(dist[pair_index])

    for d in dist_ls:
        exp_sum += np.exp(-d**2 / k)

    for j in range(len(dist_ls)):
        if i != j:
            if j > i:
                p[j+1] = np.exp(-dist_ls[j]**2 / k) / exp_sum
            else:
                p[j] = np.exp(-dist_ls[j]**2 / k) / exp_sum

    return p


def sym(p):
    start_time = time.time()
    n = np.size(p, 0)
    pair_ls = [pair for pair in combinations(range(n), 2)]
    no_of_pairs = len(pair_ls)
    sym_p = np.zeros(no_of_pairs)

    for c, pair in zip(range(no_of_pairs), pair_ls):
        sym_p[c] = (p[pair[0], pair[1]] + p[pair[1], pair[0]]) / (2*n)

    print('symmetrized p : {} seconds'.format(time.time() - start_time))
    return sym_p


def t_SNE(x, perplexity, iteration, lr, alpha):
    n = np.size(x, 0)
    p = pairwise_prob(x, perplexity)
    sym_p = sym(p)
    y = [np.random.normal(0, 0.1, (n, 2))]
    for t in range(iteration):
        # q = lowdim_prob(y)
        # gradient = get_grad(p, q, y)
        gradient = np.ones((n, 2))
        if t < 2:
            y.append(y[-1] + lr * gradient)
        else:
            y.append(y[-1] + lr*gradient + alpha*(y[-1]-y[-2]))
            y.pop(0)

    return y[-1]


iris = datasets.load_iris()
data = iris.data
print(t_SNE(data, 10, 10, 0.1, 0.2))
