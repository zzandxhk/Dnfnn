import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import tqdm
import random
from matplotlib import ticker
# from jupyterthemes import jtplot
# jtplot.style()


GRAPH_WIDTH = 10 # for a lattice graph
GRAPH_SIZE = 100 # 3x3
INPUT_DIM = 3
DATA_PER_AGENT = 10 # m/n
# TAU = 5.0 # 4 + 1
LBD = 0
CONVEX_MU = 5 # convex constant
L = 5000

OMG = 0 + 0.01 * np.random.randn(INPUT_DIM)  # normal distribution
OMG = OMG * 1e-6

MATRIX_A_COLLECTION = []
VECTOR_B_COLLECTION = []

for i in range(GRAPH_SIZE):
    A_i = np.random.rand(INPUT_DIM, DATA_PER_AGENT)  # uniform distribution
    for j in range(INPUT_DIM):  # normalize with l2 norm
        norm = np.sqrt(np.sum(np.square((A_i[j])))) # 矩阵某一行平方和开根号
        A_i[j] /= norm #数据标准化

    b_i = np.dot(A_i.T, OMG) #做内积
    # b_i = np.zeros(DATA_PER_AGENT) # force b_i = 0
    MATRIX_A_COLLECTION.append(A_i)
    VECTOR_B_COLLECTION.append(b_i)


# def OBJECTIVE_F(x):
#     res = 0
#     for i in range(GRAPH_SIZE):
#         A_i, b_i = MATRIX_A_COLLECTION[i], VECTOR_B_COLLECTION[i]
#         res += LBD * np.max(np.abs(x))  # h_i(x)
#         res += .5 * CONVEX_MU * np.sum(np.square(x))
#         res += .5 * np.sum(np.square(-b_i + np.dot(A_i.T, x)))
#         # res += .5*np.sum(np.square(np.dot(A_i.T, x)))
#     return res / GRAPH_SIZE
G = nx.grid_2d_graph(GRAPH_WIDTH, GRAPH_WIDTH)
# nx.draw(G)
# G = nx.Graph()          # 建立无向图
# H = nx.path_graph(GRAPH_SIZE)  # 添加节点
# G.add_nodes_from(H)     # 添加节点
# def rand_edge(vi,vj,p=0.3):      #默认概率p=0.3
#     probability =random.random() #生成随机小数
#     if(probability<p):           #如果小于p
#         G.add_edge(vi,vj)        #连接vi和vj节点
# i=0                              # 添加边
# while (i<GRAPH_SIZE):
#     j=0
#     while(j<i):
#             rand_edge(i,j)       #调用rand_edge()
#             j += 1
#     i += 1
# nx.draw(G)
plt.show()

nx.degree(G)
DVweight = G.degree()
degree_max = max(span for n, span in DVweight)  #节点最大度数
TAU = degree_max + 1
W = np.eye(GRAPH_SIZE) - nx.laplacian_matrix(G)/TAU

a = np.linalg.svd(W, compute_uv=False, hermitian=True)[1] # second singular value

ETA = (1-a)*a*2/(50*50)

def prox(z):
    res = z.copy()
    for i in range(z.shape[0]):
        if z[i] > ETA*LBD:
            res[i] = z[i] - ETA*LBD
        elif z[i] < -ETA*LBD:
            res[i] = z[i] + ETA*LBD
        else:
            res[i] = 0
    return res

def G_eta_i(x, i):
    A_i, b_i = MATRIX_A_COLLECTION[i], VECTOR_B_COLLECTION[i]
    # A_i = MATRIX_A_COLLECTION[i]
    grad_f_i = CONVEX_MU*x + np.dot(A_i, np.dot(A_i.T, x)) - np.dot(A_i, b_i)
    res = 1/ETA*(x - prox(x-ETA*grad_f_i))
    return res

alpha = np.sqrt(CONVEX_MU*ETA)
eps = 1e-12
obj_F_min = 0
episodes = 3000

init_x_1 = np.random.rand(GRAPH_SIZE, INPUT_DIM)*0.3
init_x_4 = init_x_1.copy()
init_x_5 = init_x_1.copy()
init_x_6 = init_x_1.copy()
init_x_7 = init_x_1.copy()
# init_y = np.random.rand(GRAPH_SIZE, INPUT_DIM)*0.1
init_y_1 = init_x_1.copy()
init_y_6 = init_y_1.copy()
# init_v = np.random.rand(GRAPH_SIZE, INPUT_DIM)*0.1
init_v_1 = init_x_1.copy()
init_s_1 = np.asarray([G_eta_i(init_y_1[i], i) for i in range(GRAPH_SIZE)])
init_nabla_1 = init_s_1.copy()
init_xbar_1 = np.mean(init_x_1, axis=0)
init_xbar_4 = np.mean(init_x_4, axis=0)
init_xbar_5 = np.mean(init_x_5, axis=0)
init_xbar_6 = np.mean(init_x_6, axis=0)
init_xbar_7 = np.mean(init_x_7, axis=0)

init_x_2 = np.random.rand(1, INPUT_DIM)*0.3
init_x_3 = init_x_2


# track_log_1 = np.zeros(episodes)
# for i in tqdm.trange(episodes):
#     track_log_1[i] = OBJECTIVE_F(init_xbar_1)
#
#     if OBJECTIVE_F(init_xbar_1) - obj_F_min < eps:
#         break
#     next_x_1 = np.asarray(W @ init_y_1 - ETA * init_s_1)
#     next_v_1 = np.asarray((1 - alpha) * W @ init_v_1 + alpha * W @ init_y_1 - ETA / alpha * init_s_1)
#     next_y_1 = np.asarray((next_x_1 + alpha * next_v_1) / (1 + alpha))
#     next_nabla_1 = np.asarray([G_eta_i(next_y_1[i], i) for i in range(GRAPH_SIZE)])
#     next_s_1 = np.asarray(W @ init_s_1 + next_nabla_1 - init_nabla_1)
#
#     init_x_1 = next_x_1
#     init_v_1 = next_v_1
#     init_y_1 = next_y_1
#     init_s_1 = next_s_1
#     init_nabla_1 = next_nabla_1
#     init_xbar_1 = np.mean(init_x_1, axis=0)
#
# mpl.rcParams.update({'font.size':10, 'axes.linewidth': 1,
#                      'xtick.major.width': 1, 'xtick.minor.width': 1,
#                      'ytick.major.width': 1, 'ytick.minor.width': 1,
#                      'xtick.major.size' : 4, 'xtick.minor.size' : 2,
#                      'ytick.major.size' : 4, 'ytick.minor.size' : 2,
#                      'xtick.top':  True, 'ytick.right': True,
#                      'xtick.direction': 'in', 'ytick.direction': 'in'})
#
# fig, ax = plt.subplots()
#
# ax.xaxis.set_major_locator(ticker.AutoLocator())
# ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
# ax.yaxis.set_major_locator(ticker.AutoLocator())
# ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
#
# ax.set_yscale('log')
#
# ax.set_xlim(0, episodes+1)    #设置横轴范围
# ax.set_ylim(1e-12, 1e-0)    #设置纵轴范围
#
# ax.set_title('Convergence graph', fontsize=15, fontweight=800)
# ax.set_xlabel('Iterations', fontsize=13, fontweight=400)
# ax.set_ylabel('(Average) Objective Error', fontsize=13, fontweight=400)
# ax.tick_params(axis='both', which='major', labelsize=9)
#
# ln1, = ax.plot(track_log_1, label='Acc-DCNGD')
# plt.legend(handles=[ln1], loc='lower right')
# plt.show()



# 算法一

OMG = 0 + 0.01*np.random.randn(INPUT_DIM) # normal distribution
OMG = OMG * 1e-6

A = np.random.rand(INPUT_DIM, DATA_PER_AGENT) # uniform distribution
for j in range(INPUT_DIM): # normalize with l2 norm
    norm = np.sqrt(np.sum(np.square((A[j]))))
    A[j] /= norm
b = np.dot(A.T, OMG)

def OBJECTIVE_F(x):
    res = 0
    res += .5*CONVEX_MU*np.sum(np.square(x))
    res += .5*np.sum(np.square(-b + np.dot(x, A)))
    return res

def grad_F(x):
    return CONVEX_MU*x + np.dot(x, A @ A.T) - np.dot(A, b)
# init_x_3 = np.random.rand(1, INPUT_DIM)*0.1
init_v_3 = init_x_3.copy()
init_y_3 = init_x_3.copy()
ETA_1 = 0.049/L
alpha = np.sqrt(CONVEX_MU*ETA)

track_log_3 = np.zeros(episodes)
for i in tqdm.trange(episodes):
    track_log_3[i] = OBJECTIVE_F(init_x_3)

    if OBJECTIVE_F(init_x_3) - obj_F_min < eps:
        break
    next_x_3 = init_y_3 - ETA * grad_F(init_y_3)
    next_v_3 = (1 - alpha) * init_v_3 + alpha * init_y_3 - ETA_1 / alpha * grad_F(init_y_3)
    next_y_3 = (next_x_3 + alpha * next_v_3) / (1 + alpha)

    init_x_3 = next_x_3
    init_v_3 = next_v_3
    init_y_3 = next_y_3

# 算法四



init_x_5 = init_x_2.copy()


OMG = 0 + 0.01*np.random.randn(INPUT_DIM) # normal distribution
OMG = OMG * 1e-6

A = np.random.rand(INPUT_DIM, DATA_PER_AGENT) # uniform distribution
for j in range(INPUT_DIM): # normalize with l2 norm
    norm = np.sqrt(np.sum(np.square((A[j]))))
    A[j] /= norm
b = np.dot(A.T, OMG)

def OBJECTIVE_F(x):
    res = 0
    res += .5*CONVEX_MU*np.sum(np.square(x))
    res += .5*np.sum(np.square(-b + np.dot(x, A)))
    return res

def grad_F(x):
    return CONVEX_MU*x + np.dot(x, A @ A.T) - np.dot(A, b)
# init_x_3 = np.random.rand(1, INPUT_DIM)*0.1
init_v_3 = init_x_5.copy()
init_y_3 = init_x_5.copy()

ETA_1 = 0.0397/L
alpha = np.sqrt(CONVEX_MU*ETA)

track_log_6 = np.zeros(episodes)
for i in tqdm.trange(episodes):
    track_log_6[i] = OBJECTIVE_F(init_x_5)

    if OBJECTIVE_F(init_x_5) - obj_F_min < eps:
        break
    next_x_3 = init_y_3 - ETA * grad_F(init_y_3)
    next_v_3 = (1 - alpha) * init_v_3 + alpha * init_y_3 - ETA_1 / alpha * grad_F(init_y_3)
    next_y_3 = (next_x_3 + alpha * next_v_3) / (1 + alpha)

    init_x_5 = next_x_3
    init_v_3 = next_v_3
    init_y_3 = next_y_3


# 算法二

init_x_4 = init_x_2.copy()
OMG = 0 + 0.01*np.random.randn(INPUT_DIM) # normal distribution
OMG = OMG * 1e-6

A = np.random.rand(INPUT_DIM, DATA_PER_AGENT) # uniform distribution
for j in range(INPUT_DIM): # normalize with l2 norm
    norm = np.sqrt(np.sum(np.square((A[j]))))
    A[j] /= norm
b = np.dot(A.T, OMG)

def OBJECTIVE_F(x):
    res = 0
    res += .5*CONVEX_MU*np.sum(np.square(x))
    res += .5*np.sum(np.square(-b + np.dot(x, A)))
    return res

def grad_F(x):
    return CONVEX_MU*x + np.dot(x, A @ A.T) - np.dot(A, b)
# init_x_3 = np.random.rand(1, INPUT_DIM)*0.1
init_v_3 = init_x_4.copy()
init_y_3 = init_x_4.copy()

ETA_1 = 0.0616/L
alpha = np.sqrt(CONVEX_MU*ETA)

track_log_5 = np.zeros(episodes)
for i in tqdm.trange(episodes):
    track_log_5[i] = OBJECTIVE_F(init_x_4)

    if OBJECTIVE_F(init_x_4) - obj_F_min < eps:
        break
    next_x_3 = init_y_3 - ETA * grad_F(init_y_3)
    next_v_3 = (1 - alpha) * init_v_3 + alpha * init_y_3 - ETA_1 / alpha * grad_F(init_y_3)
    next_y_3 = (next_x_3 + alpha * next_v_3) / (1 + alpha)

    init_x_4 = next_x_3
    init_v_3 = next_v_3
    init_y_3 = next_y_3



# 算法三

OMG = 0 + 0.01*np.random.randn(INPUT_DIM) # normal distribution
OMG = OMG * 1e-6

A = np.random.rand(INPUT_DIM, DATA_PER_AGENT) # uniform distribution
for j in range(INPUT_DIM): # normalize with l2 norm
    norm = np.sqrt(np.sum(np.square((A[j]))))
    A[j] /= norm
b = np.dot(A.T, OMG)

def OBJECTIVE_F(x):
    res = 0
    res += .5*CONVEX_MU*np.sum(np.square(x))
    res += .5*np.sum(np.square(-b + np.dot(x, A)))
    return res

def grad_F(x):
    return CONVEX_MU*x + np.dot(x, A @ A.T) - np.dot(A, b)
# init_x_3 = np.random.rand(1, INPUT_DIM)*0.1
init_v_3 = init_x_2.copy()
init_y_3 = init_x_2.copy()

ETA_1 = 0.0773/L
alpha = np.sqrt(CONVEX_MU*ETA)

track_log_4 = np.zeros(episodes)
for i in tqdm.trange(episodes):
    track_log_4[i] = OBJECTIVE_F(init_x_2)

    if OBJECTIVE_F(init_x_2) - obj_F_min < eps:
        break
    next_x_3 = init_y_3 - ETA * grad_F(init_y_3)
    next_v_3 = (1 - alpha) * init_v_3 + alpha * init_y_3 - ETA_1 / alpha * grad_F(init_y_3)
    next_y_3 = (next_x_3 + alpha * next_v_3) / (1 + alpha)

    init_x_2 = next_x_3
    init_v_3 = next_v_3
    init_y_3 = next_y_3









fig, ax = plt.subplots()

ax.set_yscale('log')

ax.set_xlim(0, episodes+1)    #设置横轴范围
ax.set_ylim(1e-12, 1e-0)    #设置纵轴范围

ax.set_title('Convergence graph', fontweight=800)
ax.set_xlabel('Iterations', fontweight=400)
ax.set_ylabel('(Average) Objective Error', fontweight=400)
ax.tick_params(axis='both', which='major', labelsize=14)

ln2, = ax.plot(track_log_5, label='N=150', marker='o', markersize=6, markevery=100)
ln1, = ax.plot(track_log_3, label='N=300', marker='s', markersize=6, markevery=100)
ln4, = ax.plot(track_log_6, label='N=400', marker='v', markersize=6, markevery=100)
ln3, = ax.plot(track_log_4, label='N=80', marker='+', markersize=8, markevery=100)

plt.legend(handles=[ln2, ln1, ln4, ln3], loc='upper right')
plt.show()
