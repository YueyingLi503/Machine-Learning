import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import *
px = pd.read_csv('PerceptronX.csv',header = None)
px = px.to_numpy()
py = pd.read_csv('Perceptrony.csv',header = None)
py = py.to_numpy()
def perceptron(X, y, max_iter=100):
    w = np.zeros(X.shape[1])
    for nmb_iter in range(1,max_iter+1):
        l =[]
        for i in range(len(X)):
            xi = X[i]
            yi = y[i]
            pred = 0
            if np.dot(w,xi.T) > 0:
                pred = 1
            else:
                pred = -1
            ans = yi*pred
            if(ans <= 0):
                l.append([i,xi,yi])
        np.random.seed(1)
        if len(l) ==0:
            break 
        else:
            i = np.random.randint(0,len(l))
            w = w+(l[i][2]*l[i][1])
    return w, nmb_iter
w, nmb_iter = perceptron(px, py, 100)
fig, ax = plt.subplots()
plot_perceptron(ax, px, py, w) # your implementation
ax.set_title(f"w={w}, iterations={nmb_iter}")
plt.savefig("name.png", dpi=300) # if you want to save your plot as a png
plt.show()


def perceptron2(X, y, max_iter=100):
    np.random.seed(1)
    alpha = np.zeros(X.shape[0])
    for nmb_iter in range(1,max_iter+1):
        mis_index = []
        #find sum of yj alpha_j X_j
        sumj = np.zeros(X.shape[1])
        for j in range(X.shape[0]):
            re = y[j] * alpha[j]*X[j]
            sumj = np.add(sumj, re)
        for i in range(len(X)):
            pred = 0
            sumxt = np.dot(sumj,X[i].T)
            if sumxt > 0:
                pred = 1
            else:
                pred = -1
            if y[i] * pred <= 0:
                mis_index.append(i)
        if len(mis_index) == 0:
            break
        else:
            idx = np.random.randint(len(mis_index), size = 1)
            alpha[mis_index[idx[0]]] += 1
    return alpha, nmb_iter
alpha, nmb_iter = perceptron2(px, py, 100)
#print(alpha)
w = np.zeros(px.shape[1])
for i in range(82):
    w += alpha[i]*py[i]*px[i]
fig, ax = plt.subplots()
plot_perceptron(ax, px, py, w) # your implementation
ax.set_title(f"w={w}, iterations={nmb_iter}")
plt.savefig("name2.png", dpi=300) # if you want to save your plot as a png
plt.show()
plt.plot(alpha)

def perceptronI(X, y, max_iter=100):
    w = np.zeros(px.shape[1])
    I = np.zeros(px.shape[0])
    for nmb_iter in range(max_iter):
        l =[]
        for i in range(len(X)):
            xi = X[i]
            yi = y[i]
            
            ans = yi*np.dot(w,xi.T)+(I[i]*2)
            if(ans <= 0):
                l.append([i,xi,yi])
        np.random.seed(1)
        if len(l) ==0:
            break 
        else:
            i = np.random.randint(0,len(l))
            I[l[i][0]] = 1
            w = w+(l[i][2]*l[i][1])
    return w, nmb_iter
w, nmb_iter = perceptronI(px, py, 100)
fig, ax = plt.subplots()
plot_perceptron(ax, px, py, w) # your implementation
ax.set_title(f"w={w}, iterations={nmb_iter}")
plt.savefig("name3.png", dpi=300) # if you want to save your plot as a png
plt.show()


def perceptron4(X, y, max_iter=100):
    np.random.seed(1)
    alpha = np.zeros(X.shape[0])
    I = np.zeros(X.shape[0])
    for nmb_iter in range(1,max_iter+1):
        mis_index = []
        #find sum of yj alpha_j X_j
        sumj = np.zeros(X.shape[1])
        for j in range(X.shape[0]):
            re = y[j] * alpha[j]*X[j]
            sumj = np.add(sumj, re)
        for i in range(len(X)):
            sumxt = np.dot(sumj, X[i].T)
            if sumxt > 0:
                pred = 1
            else:
                pred = -1
            if y[i] * pred + (I[i]*2) <= 0:
                mis_index.append(i)
        if len(mis_index) == 0:
            break
        else:
            idx = np.random.randint(len(mis_index), size = 1)
            alpha[mis_index[idx[0]]] += 1
            I[mis_index[idx[0]]] = 1
    return alpha, nmb_iter
alpha, nmb_iter = perceptron4(px, py, 100)
w = np.zeros(px.shape[1])
for i in range(82):
    w += alpha[i]*py[i]*px[i]
fig, ax = plt.subplots()
plot_perceptron(ax, px, py, w) # your implementation
ax.set_title(f"w={w}, iterations={nmb_iter}")
plt.savefig("name4.png", dpi=300) # if you want to save your plot as a png
plt.show()