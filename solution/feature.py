import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
Cs = np.linspace(0.001, 0.2, num=100)
l = []
clog = []
for c in Cs:
    clog.append(math.log(c))
    clf = LogisticRegression(penalty='l1', solver='liblinear', C=c)
    clf.fit(X_train, y_train.ravel())
    l.append(clf.coef_[0])
dffeature = pd.DataFrame(l, columns = X.columns)
#print(dffeature)
plt.plot(clog, dffeature['Popularity'], color = 'red')
plt.plot(clog, dffeature['danceability'], color = 'brown')
plt.plot(clog, dffeature['energy'], color = 'green')
plt.plot(clog, dffeature['loudness'], color = 'blue')
plt.plot(clog, dffeature['speechiness'], color = 'orange')
plt.plot(clog, dffeature['acousticness'], color = 'pink')
plt.plot(clog, dffeature['liveness'], color = 'purple')
plt.plot(clog, dffeature['valence'], color = 'grey')
plt.plot(clog, dffeature['tempo'], color = 'black')
plt.plot(clog, dffeature['duration_in min/ms'], color = 'y')
plt.show()