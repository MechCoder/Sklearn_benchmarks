from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model.tests.test_sparse_coordinate_descent import make_sparse_data
from time import time
import pylab as pl 
import numpy as np

X, y = make_sparse_data(n_samples=500, n_features=2000, n_informative=200)
n_cores = [1, 2, 4]
n_alpha = [5, 10, 50, 100]
times = [0] * 12

counter = 0
for _ in range(3):
    for core in n_cores:
        for alpha in n_alpha:
		    clf = ElasticNetCV(n_jobs=core, n_alphas=alpha,
		        	           l1_ratio=0.5, cv=10)
		    print "core = %d, alpha = %d" % (core, alpha)
		    t = time()
		    clf.fit(X, y)
		    times[counter%12] += (time() - t)
		    print times
		    counter += 1

# Got after doing the above. Just for future reference.
core1_mp = [57.457534631093345, 72.31527137756348, 210.2204163869222, 379.9918119907379]
core2_mp = [55.89718206723531, 51.196732918421425, 138.35079900423685, 239.67310031255087]
core3_mp = [42.53018967310587, 49.97517212231954, 122.26631005605061, 204.76643363634744]

core1_t = [60.99967805544535, 75.41305232048035, 219.61244002978006, 390.601344982783]
core2_t = [46.21716833114624, 54.701584259668984, 144.06910300254822, 242.6696043809255]
core3_t = [43.21849703788757, 49.07820804913839, 122.74103697141011, 205.75086871782938]

_, [axis1, axis2, axis3] = pl.subplots(3, 1, sharex=True)
ind = np.arange(4)
width = 0.35

axis1.set_title("n_jobs = 1, Multiprocessing vs threading")
bar1m = axis1.bar(ind, core1_mp, width, color="r")
bar2m = axis1.bar(ind + width, core1_t, width, color="y")
axis1.set_ylabel("Time")
axis1.set_xticks(ind + width)
axis1.set_xticklabels(('5', '10', '50', '100'))
axis1.legend((bar1m[0], bar2m[0]), ("In this branch", "In master"), loc="upper left")

axis2.set_title("n_jobs = 2")
bar1m = axis2.bar(ind, core2_mp, width, color="r")
bar2m = axis2.bar(ind + width, core2_t, width, color="y")
axis2.set_ylabel("Time")
axis2.set_xticks(ind + width)
axis2.set_xticklabels(('5', '10', '50', '100'))
axis2.legend((bar1m[0], bar2m[0]), ("In this branch", "In master"), loc="upper left")

axis3.set_title("n_jobs = 4")
bar1m = axis3.bar(ind, core3_mp, width, color="r")
bar2m = axis3.bar(ind + width, core3_t, width, color="y")
axis3.set_ylabel("Time")
axis3.set_xticks(ind + width)
axis3.set_xticklabels(('5', '10', '50', '100'))
axis3.legend((bar1m[0], bar2m[0]), ("In this branch", "In master"), loc="upper left")
pl.savefig("Sparse.png")