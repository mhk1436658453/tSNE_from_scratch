import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
import matplotlib.animation as animation
from sklearn import manifold
from sklearn import datasets
from scipy.stats import entropy
from scipy.spatial.distance import pdist
from itertools import combinations
from math import isnan


def pairwise_prob(x, perplexity):
    sq_dist = pdist(x, 'sqeuclidean')
    n = np.size(x, 0)
    pair_ls = [pair for pair in combinations(range(n), 2)]

    p = np.zeros((n, n))
    start_time = time.time()
    for i in range(n):

        sigma = 0.1
        tolerance = 0.01
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

    print('compute p given i: {} seconds'.format(time.time() - start_time))
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
    start_time = time.time()
    n = np.size(p, 0)
    pair_ls = [pair for pair in combinations(range(n), 2)]
    no_of_pairs = len(pair_ls)
    sym_p = np.zeros(no_of_pairs)

    for c, pair in zip(range(no_of_pairs), pair_ls):
        sym_p[c] = (p[pair[0], pair[1]] + p[pair[1], pair[0]]) / (2*n)

    print('compute symmetrized p : {} seconds'.format(time.time() - start_time))
    return sym_p


def lowdim_prob(y, sq_dist):
    #start_time = time.time()
    dist_sum = sq_dist.sum()*2

    q = np.zeros(sq_dist.size)

    for c, d in zip(range(sq_dist.size), sq_dist):
        q[c] = d/dist_sum

    #print('compute q: {} seconds'.format(time.time() - start_time))
    return q


def get_grad(y, p, q, d, pr_ls):
    start_time = time.time()
    g = np.zeros(y.shape)
    n = np.size(y, 0)
    p_q = p-q
    for i in range(n):
        temp_g = 0
        #compute_time = time.time()
        for j in range(n):

            if i != j:
                if i > j:
                    index = pr_ls.index((j, i))
                else:
                    index = pr_ls.index((i, j))

                temp_g += -4*(p_q[index])*(y[i]-y[j])*(d[index])

        g[i] = temp_g
        #print('compute: {} seconds'.format(time.time() - compute_time))
    print('compute gradient: {} seconds'.format(time.time() - start_time))
    return g


def t_SNE(x, perplexity, iteration, lr, alpha):
    n = np.size(x, 0)
    pr_ls = [pair for pair in combinations(range(n), 2)]
    pair_p = pairwise_prob(x, perplexity)
    p = sym(pair_p)

    y = [np.random.normal(0, 0.1, (n, 2))]
    fig_vis = plt.figure(1)
    ax_vis = fig_vis.add_subplot(1, 1, 1)
    # ax_vis.scatter(y[-1][red, 0], y[-1][red, 1], c="r")
    # ax_vis.scatter(y[-1][green, 0], y[-1][green, 1], c="g")
    ax_vis.scatter(y[-1][:, 0], y[-1][:, 1], c=label)
    ax_vis.legend(label_names)

    plt.pause(0.3)
    for t in range(iteration):
        print('iteration', t)
        sq_dist = 1/(1+pdist(y[-1], 'sqeuclidean'))
        q = lowdim_prob(y[-1], sq_dist)

        gradient = get_grad(y[-1], p, q, sq_dist, pr_ls)
        #gradient = np.ones((n, 2))

        if t < 2:
            y.append(y[-1] + lr * gradient)
        else:
            y.append(y[-1] + lr*gradient + alpha*(y[-2]-y[-3]))
            y.pop(0)

        ax_vis.cla()

        # ax_vis.scatter(y[-1][red, 0], y[-1][red, 1], c="r")
        # ax_vis.scatter(y[-1][green, 0], y[-1][green, 1], c="g")
        ax_vis.scatter(y[-1][:, 0], y[-1][:, 1], c=label)
        ax_vis.legend(label_names)

        plt.pause(0.3)
        print("kl_divergence {}".format(entropy(p, q)))
    return y[-1]


iris = datasets.load_iris()
data = iris.data
label = iris.target
label_names = list(iris.target_names)
# X, y_ = datasets.make_circles(n_samples=150, factor=.5, noise=.02)
# red = y_ == 0
# green = y_ == 1


# plt.figure()
# plt.scatter(X[red, 0], X[red, 1], c="r")
# plt.scatter(X[green, 0], X[green, 1], c="g")
# plt.show()

# tsne = manifold.TSNE(n_components=2, init='random',
#                          random_state=0, perplexity=5)
# Y = tsne.fit_transform(X)
#
# plt.scatter(Y[red, 0], Y[red, 1], c="r")
# plt.scatter(Y[green, 0], Y[green, 1], c="g")
# plt.show()




# t_SNE(x, perplexity, iteration, learning rate, alpha)
dr = (t_SNE(data, 30, 40, 70, 0.2))


plt.figure(2)
plt.scatter(dr[:, 0], dr[:, 1], c=label)
plt.show()
