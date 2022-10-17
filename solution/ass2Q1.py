import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics 
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import math

df = pd.read_csv('songs.csv')
df = df.drop(['Artist Name','Track Name','key','mode','time_signature','instrumentalness'],axis = 1)
df = df[(df['Class']==5)|(df['Class']==9)]
df = df.replace({5:1,9:-1})
df = df.dropna()
y = df['Class'].to_frame()
X = df.drop(['Class'],axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 23)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
y_train = y_train.to_numpy()
def reg_log_loss(W, C, X, y):
    W = pd.DataFrame(W)
    X_temp  = np.insert(X, 0, values = 1,axis = 1)
    xw = np.dot(X_temp, W)
    yxw = -y * xw
    right = C * np.sum(np.logaddexp(0, yxw))
    left = np.linalg.norm(w,ord = 2)
    left = 0.5 * (left**2)
    ans = left + right
    #print(right)
    return ans
c = -1.4
c2 = 1.2
w = 0.1 * np.ones(X_train.shape[1])
w2 = 0.35 * np.ones(X_train.shape[1])
W = np.insert(w, 0, c)
W2 = np.insert(w2, 0, c2)
ans = reg_log_loss(W,0.001, X_train, y_train)
ans2 = reg_log_loss(W2,0.001, X_train, y_train)
print(ans)
print(ans2)

w = 0.1 * np.ones(X_train.shape[1])
#print(w.shape)
W0 = np.insert(w,0, -1.4)
C= 0.4
#print(reg_log_loss(W0, C, X_train, y_train))
def reg_los_fit(X, y ,C):
    args = (C, X, y)
    g = lambda x: reg_log_loss(x, *args)
    #print(g(W0))
    res = minimize(g, W0, method='Nelder-Mead',options = {'maxiter':9999}, tol=1e-6)
    return (res.x)

X_temp  = np.insert(X_train, 0, values = 1,axis = 1)
W_hat = reg_los_fit(X_train, y_train, C)
W_hat = pd.DataFrame(W_hat)
xw_hat = np.dot(X_temp, W_hat)
pred = expit(xw_hat)
pred = pd.DataFrame(pred,columns=['Class'])
train_loss = metrics.log_loss(y_train, pred)
print('my model train loss is {a}'.format(a=train_loss))

X_temp2  = np.insert(X_test, 0, values = 1,axis = 1)
xw_hat = np.dot(X_temp2, W_hat)
pred = expit(xw_hat)
pred = pd.DataFrame(pred,columns=['Class'])
test_loss = metrics.log_loss(y_test, pred)
print('my model test loss is {a}'.format(a=test_loss))

clf = LogisticRegression(C=1, penalty='l2', tol=1e-6)
clf.fit(X_train, y_train.ravel())
y_train_pred = clf.predict_proba(X_train)
y_test_pred = clf.predict_proba(X_test)
sk_train_loss = metrics.log_loss(y_train, y_train_pred)
sk_test_loss = metrics.log_loss(y_test, y_test_pred)
print('sklearn model train loss is {a}'.format(a=sk_train_loss))
print('sklearn model test loss is {a}'.format(a=sk_test_loss))

Cs = np.linspace(0.0001, 0.8, num=25)
X_loov=X_train[0:544,:]
y_loov = y_train[0:544,:]
ansy = []
for c in Cs:
    loss = []
    for i in range(len(X_loov)):
        X_new = np.delete(X_loov,(i),axis = 0)
        y_new = np.delete(y_loov,(i),axis = 0)
        clf = LogisticRegression(penalty='l1', solver='liblinear', C=c)
        clf.fit(X_new, y_new.ravel())
        X_one_pred = clf.predict_proba(X_loov[i].reshape(1,-1))
        loss.append(metrics.log_loss(y_loov[i], X_one_pred,labels = (1,-1)))
    ansy.append(np.mean(loss))
#print(ansy)
plt.plot(Cs, ansy)
plt.xlabel('C')
plt.ylabel('Average loss for n samples')
plt.show()