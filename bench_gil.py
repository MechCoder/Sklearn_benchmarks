r"""
This benchmark was a quick and hacky script written as a method to
compare multiprocessing, threading between master (without releasing GIL)
and the branch gil-enet in my repo. This script was run three times in both
master and the gil-enet branch, and the mean taken. Pardon the reading and
writing from files in between.
"""

from collections import defaultdict

from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.datasets import make_regression
from time import time
import pylab as pl
import numpy as np
from numpy.random import RandomState

n_alphas = [5, 10, 50, 100]
times = defaultdict(list)
rng = RandomState(0)
n_l1_ratio = [1, 2, 5]
colors = ["b-", "g-", "r-", "c-", "w-"]
a = open("times_master.txt", "a+")
X, y = make_regression(n_samples=500, n_features=2000, random_state=rng)

# Test for multiple cores, multiple l1_ratios and alphas. 
for core in [1, 2, 4]:
    a.write("\n" % core)
    for l1 in n_l1_ratio:
        for i, alpha in enumerate(n_alphas):
            clf = ElasticNetCV(n_alphas=alpha,
                               l1_ratio=np.linspace(0.1, 0.9, l1),
                               cv=10, n_jobs=core)
            print "Iteration", str(core), str(alpha), str(l1)
            t = time()
            clf.fit(X, y)
            a.write(str(time() - t) + " ")
        a.write("\n")

a.write("\n")