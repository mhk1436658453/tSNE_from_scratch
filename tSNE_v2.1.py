import numpy as np
import matplotlib.pyplot as plt
import scipy
import time

from sklearn import datasets
from scipy.stats import entropy
from scipy.spatial.distance import pdist
from itertools import combinations
from math import isnan


def pairwise_prob(x, perplexity):
    sq_dist = pdist(x, 'sqeuclidean')
    n = np.size(x, 0)

    p = np.zeros((n, n))
    for i in range(n):

        sigma = 0.1
        tolerance = perplexity*0.01
        sigma_low = 0
        sigma_high = np.NaN
        while True:
            p_i = p_given_i(i, n, sq_dist, sigma)

            error = 2 ** entropy(p_i) - perplexity
            if abs(error) > tolerance:
                if error > 0:
                    sigma_high = sigma
                    sigma = (sigma_low + sigma) / 2
                else:
                    sigma_low = sigma
                    sigma = sigma * 2 if isnan(sigma_high) else (sigma_high + sigma) / 2
            else:
                break

        p[i] = p_i

    return p


# compute p_ji given i for all j
def p_given_i(i, n , sq_dist, sigma):

    p = np.zeros(n)
    pair_ls = [pair for pair in combinations(range(n), 2)]

    k = 2*sigma**2
    exp_sum = 0
    dist_ls = []
    for pair_index in range(len(pair_ls)):
        if i in pair_ls[pair_index]:
            dist_ls.append(sq_dist[pair_index])

    for d in dist_ls:
        exp_sum += np.exp(-d / k)

    for j in range(len(dist_ls)):
        if i != j:
            if j > i:
                p[j+1] = np.exp(-dist_ls[j] / k) / exp_sum
            else:
                p[j] = np.exp(-dist_ls[j] / k) / exp_sum

    return p


def sym(p):
    n = np.size(p, 0)
    pair_ls = [pair for pair in combinations(range(n), 2)]
    no_of_pairs = len(pair_ls)
    sym_p = np.zeros(no_of_pairs)

    for c, pair in zip(range(no_of_pairs), pair_ls):
        sym_p[c] = (p[pair[0], pair[1]] + p[pair[1], pair[0]]) / (2*n)

    return sym_p


def lowdim_prob(y, sq_dist):
    dist_sum = sq_dist.sum()*2
    q = np.zeros(sq_dist.size)

    for c, d in zip(range(sq_dist.size), sq_dist):
        q[c] = d/dist_sum

    return q


def get_grad(y, p, q, d, pr_ls):
    g = np.zeros(y.shape)
    n = np.size(y, 0)
    p_q = p-q

    for i in range(n):
        temp_g = 0
        for j in range(n):
            if i != j:
                if i > j:
                    index = pr_ls.index((j, i))
                else:
                    index = pr_ls.index((i, j))
                temp_g += -4*(p_q[index])*(y[i]-y[j])*(d[index])
        g[i] = temp_g

    return g


def t_SNE(x, perplexity, iteration, lr, alpha):
    n = np.size(x, 0)
    pr_ls = [pair for pair in combinations(range(n), 2)]
    kl_div = np.zeros(iteration)
    start_time = time.time()

    pair_p = pairwise_prob(x, perplexity)
    p = sym(pair_p)

    print('compute p given i: {} seconds'.format(time.time() - start_time))

    y = [np.random.normal(0, 0.1, (n, 2))]

    for t in range(iteration):
        print('iteration', t)

        start_time = time.time()

        sq_dist = 1/(1+pdist(y[-1], 'sqeuclidean'))
        q = lowdim_prob(y[-1], sq_dist)

        gradient = get_grad(y[-1], p, q, sq_dist, pr_ls)

        # gradient descent with momentum
        if t < 2:
            y.append(y[-1] + lr * gradient)
        else:
            y.append(y[-1] + lr*gradient + alpha*(y[-2]-y[-3]))
            #y.pop(0)
        kl_div[t] = entropy(p, q)
        print('kl_divergence {:.5f} \ttime taken: {:.5f} seconds '.format(kl_div[t], (time.time() - start_time)))

    return y, kl_div


iris = datasets.load_iris()
data = iris.data
label = iris.target
label_names = list(iris.target_names)

# t_SNE(x, perplexity, iteration, learning rate, momentum)
result, kl_history = (t_SNE(data, 15, 40, 70, 0.4))

fig = plt.figure(3)
plt.pause(3)
ax = fig.add_subplot(1, 1, 1)
for iterations, y_result in zip(range(np.size(data, 0)), result):
    ax.cla()
    ax.scatter(y_result[:, 0], y_result[:, 1], c=label)
    ax.set_title('iteration {}'.format(iterations))
    plt.pause(0.5)
plt.show()

fig = plt.figure(4)
ax = fig.add_subplot(1, 1, 1)
ax.plot(kl_history)
ax.set_title('kl_divergence')
plt.pause(0.5)
plt.show()
