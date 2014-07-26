import numpy as np
from sklearn.linear_model import *
import numpy as np
from time import time
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.datasets import *
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

cat = ['talk.religion.misc', 'alt.atheism']
data = fetch_20newsgroups(subset='train', categories=cat)
X, y = data.data, data.target
clf = TfidfVectorizer()
X = clf.fit_transform(X)
y[y == 0] = -1

# f = open("../Downloads/arcene/arcene_train.data")
# X = np.fromfile(f, dtype=np.float64, sep=' ')
# X = X.reshape(-1, 10000)
# f = open("../Downloads/arcene/arcene_train.labels")
# y = np.fromfile(f, dtype=np.int32, sep=' ')

random_time4 = []
random_time8 = []
cyclic_time4 = []
cyclic_time8 = []
random_iter4 = []
random_iter8 = []
cyclic_iter4 = []
cyclic_iter8 = []
random_score4 = []
random_score8 = []
cyclic_score4 = []
cyclic_score8 = []

alphas = _alpha_grid(X, y, n_alphas=20)
for alpha in alphas:

    r_time4, r_iter4, r_score4, r_time8, r_iter8, r_score8 = 0, 0, 0, 0, 0, 0
    c_time4, c_iter4, c_score4, c_time8, c_iter8, c_score8 = 0, 0, 0, 0, 0, 0

    for n_iter in [0, 1, 2]:
        X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.33, random_state=n_iter)

        clf = ElasticNet(max_iter=500000, alpha=alpha, tol=1e-4)
        print("......") + str(alpha)
        t = time()
        clf.fit(X_train, y_train)
        c_time4 += time() - t
        y_pred = np.sign(clf.predict(X_test))
        c_iter4 += clf.n_iter_
        c_score4 += accuracy_score(y_test, y_pred)
        print c_iter4
        print c_time4
        print c_score4

        clf = ElasticNet(max_iter=500000, alpha=alpha, tol=1e-4, random_state=0, selection='random')
        print("......") + str(alpha)
        t = time()
        clf.fit(X_train, y_train)
        r_time4 += time() - t
        y_pred = np.sign(clf.predict(X_test))
        r_iter4 += clf.n_iter_
        r_score4 += accuracy_score(y_test, y_pred)
        print r_iter4
        print r_time4
        print r_score4


        clf = ElasticNet(max_iter=500000, alpha=alpha, tol=1e-8)
        print("......") + str(alpha)
        t = time()
        clf.fit(X_train, y_train)
        c_time8 += time() - t
        y_pred = np.sign(clf.predict(X_test))
        c_iter8 += clf.n_iter_
        c_score8 += accuracy_score(y_test, y_pred)
        print c_iter8
        print c_time8
        print c_score8

        clf = ElasticNet(max_iter=500000, alpha=alpha, tol=1e-8, random_state=0, selection='random')
        print("......") + str(alpha)
        t = time()
        clf.fit(X_train, y_train)
        r_time8 += time() - t
        y_pred = np.sign(clf.predict(X_test))
        r_iter8 += clf.n_iter_
        r_score8 += accuracy_score(y_test, y_pred)
        print r_iter8
        print r_time8
        print r_score8

    random_time4.append(r_time4 / 3.)
    #random_time8.append(r_time8 / 3.)
    cyclic_time4.append(c_time4 / 3.)
    #cyclic_time8.append(c_time8 / 3.)
    random_iter4.append(r_iter4 / 3.)
    #random_iter8.append(r_iter8 / 3.)
    cyclic_iter4.append(c_iter4 / 3.)
    #cyclic_iter8.append(c_iter8 / 3.)
    random_score4.append(r_score4 / 3.)
    #random_score8.append(r_score8 / 3.)
    cyclic_score4.append(c_score4 / 3.)
    #SSScyclic_score8.append(c_score8 / 3.)
    print random_score8
    print random_score4
    print cyclic_score8
    print cyclic_score4


plt.clf()
plt.subplot(311)
plt.title("alphas vs n_iters (arcene dataset)")
random_ = plt.semilogx(alphas, random_iter4, "b-")[0]
normal = plt.semilogx(alphas, cyclic_iter4, "r--")[0]
#plt.legend([random_, normal], ["Random descent", "Normal descent"], loc=1)
plt.xlim(xmin=np.min(alphas), xmax=np.max(alphas))
plt.ylim(0, max(np.max(random_iter4), np.max(cyclic_iter4)))
plt.ylabel('No of iterations. ')
#plt.xlabel('Alphas.')

plt.subplot(312)
plt.title("alphas vs time (arcene dataset),")
random_ = plt.semilogx(alphas, random_time4, "b-")[0]
normal = plt.semilogx(alphas, cyclic_time4, "r--")[0]
plt.ylabel('Time taken to converge')
    #plt.legend([random_, normal], ["Random descent", "Normal descent"], loc=1)
plt.xlim(xmin=np.min(alphas), xmax=np.max(alphas))
plt.ylim((0, max(max(random_time4), max(cyclic_time4))))
plt.ylabel('Time')
#plt.xlabel('Alphas.')

plt.subplot(313)
plt.title("alphas vs score (arcene dataset)")
random_ = plt.semilogx(alphas, random_score4, "b-")[0]
normal = plt.semilogx(alphas, cyclic_score4, "r--")[0]
    #plt.legend([random_, normal], ["Random descent", "Normal descent"], loc=1)
plt.xlabel('Grid of alphas.')
plt.ylabel('Scores obtained.')
plt.xlim(xmin=np.min(alphas), xmax=np.max(alphas))
plt.ylim((0.5, 1))
plt.show()
