import numpy as np
import matplotlib.pyplot as plt
import scipy
import time

from sklearn import datasets
from scipy.stats import entropy
from scipy.spatial.distance import pdist
from itertools import combinations
from math import isnan


def pairwise_prob(x_, perp):
    sq_dist = pdist(x_, 'sqeuclidean')
    n = np.size(x_, 0)

    p = np.zeros((n, n))
    for i in range(n):

        sigma = 0.1
        tolerance = perp*0.01
        sigma_low = 0
        sigma_high = np.NaN
        while True:
            p_i = p_given_i(i, n, sq_dist, sigma)

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


def lowdim_prob(y_, sq_dist):
    dist_sum = sq_dist.sum()*2
    q = np.zeros(sq_dist.size)

    for c, d in zip(range(sq_dist.size), sq_dist):
        q[c] = d/dist_sum

    return q


def get_grad(y_, p, q, d, pr_ls):
    g = np.zeros(y_.shape)
    n = np.size(y_, 0)
    p_q = p-q

    for i in range(n):
        temp_g = 0
        for j in range(n):
            if i != j:
                if i > j:
                    index = pr_ls.index((j, i))
                else:
                    index = pr_ls.index((i, j))
                temp_g += -4*(p_q[index])*(y_[i]-y_[j])*(d[index])
        g[i] = temp_g

    return g


def t_SNE(x_, perp, iteration, lr, alpha):
    n = np.size(x_, 0)
    pr_ls = [pair for pair in combinations(range(n), 2)]
    kl_div = np.zeros(iteration+1)
    start_time = time.time()

    pair_p = pairwise_prob(x_, perp)
    p = sym(pair_p)

    print('compute p given i: {} seconds'.format(time.time() - start_time))

    y_ = [np.random.normal(0, 0.1, (n, 2))]

    for t in range(iteration+1):
        print('iteration', t)

        start_time = time.time()

        sq_dist = 1/(1+pdist(y_[-1], 'sqeuclidean'))
        q = lowdim_prob(y_[-1], sq_dist)

        gradient = get_grad(y_[-1], p, q, sq_dist, pr_ls)

        # gradient descent with momentum
        if t < 2:
            y_.append(y_[-1] + lr * gradient)
        else:
            y_.append(y_[-1] + lr*gradient + alpha*(y_[-2]-y_[-3]))

        kl_div[t] = entropy(p, q)
        print('kl_divergence {:.5f} \ttime taken: {:.5f} seconds '.format(kl_div[t], (time.time() - start_time)))

    return y_, kl_div


# load iris datasets
x = np.linspace(0, 1, int(np.sqrt(100)))
xx, yy = np.meshgrid(x, x)
data = np.hstack([
    xx.ravel().reshape(-1, 1),
    yy.ravel().reshape(-1, 1),
])

n_data = np.size(data, 0)
label = np.arange(n_data)

# hyper parameters
perplexity      = 15
no_of_iteration = 40
learning_rate   = 20
momentum        = 0.4

# run t-SNE
result, kl_history = (t_SNE(data, perplexity, no_of_iteration, learning_rate, momentum))

# visualize
fig, ax = plt.subplots(ncols=2, nrows=2)
gs = ax[0, 0].get_gridspec()
ax[0, 0].remove()
ax[1, 0].remove()
ax_vis = fig.add_subplot(gs[:, 0])

ax_vis.set_title('Random initialization')
ax_vis.scatter(result[0][:, 0], result[0][:, 1], c=label)

ax_kl = ax[1, 1]
ax_kl.set_title('KL divergence')

ax_real = ax[0, 1]
ax_real.scatter(data[:, 0], data[:, 1], c=label)
ax_real.set_title('Actual data')

plt.pause(4)


for iterations, y_result in zip(range(no_of_iteration), result[1:]):
    # plot t-sne iterations
    ax_vis.cla()
    ax_vis.set_title('t-SNE iteration {}'.format(iterations+1))
    ax_vis.scatter(y_result[:, 0], y_result[:, 1], c=label)

    # plot kl divergence
    ax_kl.cla()
    ax_kl.set_title('KL divergence')
    ax_kl.plot(kl_history[:iterations+1])
    ax_kl.set_ylim([0, kl_history[0]*1.1])
    ax_kl.set_xlim([0, no_of_iteration])
    ax_kl.set_ylabel('kl divergence')
    ax_kl.set_xlabel('iterations')
    plt.pause(0.3)

plt.show()
