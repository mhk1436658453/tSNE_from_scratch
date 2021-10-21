import numpy as np
import matplotlib.pyplot as plt
import scipy
import time

from decimal import *
from sklearn import datasets
from scipy.stats import entropy
from scipy.spatial.distance import pdist
from itertools import combinations
from math import isnan
from mpl_toolkits.mplot3d import Axes3D


def pairwise_prob(x_, perp, pr_map):
    sq_dist = pdist(x_, 'sqeuclidean')
    n = np.size(x_, 0)
    max_search = 50
    p = np.zeros((n, n))

    for i in range(n):
        if i % 50 == 0:
            print('computing p_ij for point {:>5} to point {:>5}...'.format(i, i+50))
        sigma = 0.1
        tolerance = perp*0.01
        sigma_low = 0
        sigma_high = np.NaN

        search = 0
        while search < max_search:
            p_i = p_given_i(i, n, sq_dist, sigma, pr_map)
            error = 2 ** entropy(p_i) - perp
            if abs(error) > tolerance:
                if error > 0:
                    sigma_high = sigma
                    sigma = (sigma_low + sigma) / 2
                else:
                    sigma_low = sigma
                    sigma = sigma * 2 if isnan(sigma_high) else (sigma_high + sigma) / 2
            else:
                break
            search += 1

        p[i] = p_i

    return p


# compute p_ji given i for all j
def p_given_i(i, n , sq_dist, sigma, pr_map):
    p = np.zeros(n)
    k = 1/(2 * sigma ** 2)
    exp_sum = 0

    dist = np.zeros(n)
    for j in range(n):
        if i != j:
            dist[j] = sq_dist[int(pr_map[i, j])]

    for d in dist:
        exp_sum += np.exp(-d * k)

    for j in range(n):
        if i != j:
            p[j] = np.exp(-dist[j] * k) / exp_sum
        else:
            j += 1

    return p


# symmetrized probability
def sym(p):
    n = np.size(p, 0)
    pair_ls = [pair for pair in combinations(range(n), 2)]
    no_of_pairs = len(pair_ls)
    sym_p = np.zeros(no_of_pairs)

    for c, pair in zip(range(no_of_pairs), pair_ls):
        sym_p[c] = (p[pair[0], pair[1]] + p[pair[1], pair[0]]) / (2*n)

    return sym_p


# probability q at low dimension space
def lowdim_prob(y_, sq_dist):

    dist_sum = sq_dist.sum()*2
    q = np.zeros(sq_dist.size)

    for c, d in zip(range(sq_dist.size), sq_dist):
        q[c] = d/dist_sum

    return q


# compute gradient
def get_grad(y_, p, q, d, pr_ls, pr_map):
    g = np.zeros(y_.shape)
    n = np.size(y_, 0)
    m = -4*(p-q)*d
    for i in range(n):
        temp_g = 0
        for j in range(n):
            if i != j:
                temp_g += m[int(pr_map[i, j])]*(y_[i]-y_[j])

        g[i] = temp_g

    return g


def t_SNE(x_, perp, iteration, lr, alpha):
    n = np.size(x_, 0)
    pr_ls = [pair for pair in combinations(range(n), 2)]
    kl_div = np.zeros(iteration+1)

    # create pair map
    # pair_map[i,j] will give the index of pair wise metrics (distance, p, q etc)
    pairmap_time = time.time()
    count = 0
    pair_map = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and i < j:
                pair_map[i, j] = count
                count += 1
    pair_map = pair_map + pair_map.T
    np.fill_diagonal(pair_map, np.NaN)
    print('Make pair map: {} seconds'.format(time.time() - pairmap_time))

    start_time = time.time()

    pair_p = pairwise_prob(x_, perp, pair_map)
    p = sym(pair_p)

    print('compute p given i: {} seconds'.format(time.time() - start_time))


    y_ = [np.random.normal(0, 0.1, (n, 2))]

    for t in range(iteration+1):
        print('iteration', t)

        start_time = time.time()

        sq_dist = 1/(1+pdist(y_[-1], 'sqeuclidean'))

        q = lowdim_prob(y_[-1], sq_dist)

        if t != iteration:
            gradient = get_grad(y_[-1], p, q, sq_dist, pr_ls, pair_map)
            # gradient descent with momentum
            if t < 2:
                y_.append(y_[-1] + lr * gradient)
            else:
                y_.append(y_[-1] + lr*gradient + alpha*(y_[-2]-y_[-3]))

        kl_div[t] = entropy(p, q)
        print('kl_divergence {:.5f} \ttime taken: {:.5f} seconds '.format(kl_div[t], (time.time() - start_time)))

    return y_, kl_div


def visualize(actual_data, y, kl, tags):
    cmap = 'viridis'

    # set up subplots
    fig, ax = plt.subplots(ncols=2, nrows=2)
    gs = ax[0, 0].get_gridspec()
    ax[0, 0].remove()
    ax[1, 0].remove()
    ax[0, 1].remove()

    # t-sne iterations
    ax_vis = fig.add_subplot(gs[:, 0])
    ax_vis.set_title('Random initialization')
    ax_vis.scatter(y[0][:, 0], y[0][:, 1], c=tags, cmap=cmap, edgecolor='k')

    # kl divergence
    ax_kl = ax[1, 1]
    ax_kl.set_title('KL divergence')

    # for 2d
    # ax_real = fig.add_subplot(gs[0, 1])
    # ax_real.scatter(data[:, 0], data[:, 1], c=label)

    # for 3d
    ax_real = fig.add_subplot(gs[0, 1], projection='3d')
    ax_real.scatter(actual_data[:, 0], actual_data[:, 1], actual_data[:, 2], c=tags, cmap=cmap, edgecolor='k')

    ax_real.set_title('Actual data')

    plt.pause(3)

    t = len(y)
    for iterations, y_result in zip(range(1, t), y[1:]):
        # t-sne iterations
        ax_vis.cla()
        ax_vis.set_title('t-SNE iteration {}'.format(iterations))
        ax_vis.scatter(y_result[:, 0], y_result[:, 1], c=tags, cmap=cmap, edgecolor='k')

        # kl divergence
        ax_kl.cla()
        ax_kl.set_title('KL divergence')
        ax_kl.plot(kl[:iterations+1])
        ax_kl.set_ylim([0, kl[0] * 1.1])
        ax_kl.set_xlim([0, no_of_iteration+1])
        ax_kl.set_ylabel('kl divergence')
        ax_kl.set_xlabel('iterations')
        if iterations > 50:
            plt.pause(0.01)
        else:
            plt.pause(0.2)

    plt.show()


# ######################################################################################################################
# make some data
# # square
# x = np.linspace(0, 1, 7)
# #xx, yy = np.meshgrid(x, x)
# xx, yy, zz = np.meshgrid(x, x, x)
# data = np.hstack([
#     xx.ravel().reshape(-1, 1),
#     yy.ravel().reshape(-1, 1),
#     zz.ravel().reshape(-1, 1),
# ])
#
# n_data = np.size(data, 0)
# label = np.arange(n_data)

iris = datasets.load_iris()
data = iris.data
label = iris.target
label_names = list(iris.target_names)




# t-SNE hyper parameters
perplexity      = 30
no_of_iteration = 200
learning_rate   = 50
momentum        = 0.2

# run t-SNE
result, kl_history = t_SNE(data, perplexity, no_of_iteration, learning_rate, momentum)

# visualize
visualize(data, result, kl_history, label)

