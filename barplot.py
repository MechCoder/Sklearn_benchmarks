import numpy as np
import pylab as pl

r"""
This assumes the timings are stored in four txt files. 
1. multi_nogil.txt
2. thread_nogil.txt
3. multi_gil.txt
4. thread_gil.txt
in the format
The benchmarks for 3 iterations, 3 cores, 3 l1_ratios and 4 alphas.
Each line corresponds to four alphas.
Each triplet of 3 lines corresponds to a core.
And 3 * 3 = 9 lines corresponds to a iteration.
"""

# Preprocessing.
m_nogil = open("multi_nogil.txt", "r")
t_nogil = open("thread_nogil.txt", "r")
m_gil = open("multi_gil.txt", "r")
t_gil = open("thread_gil.txt", "r")

iter1_ngilmp, iter2_ngilmp, iter3_ngilmp = [], [], []
iter1_ngilt, iter2_ngilt, iter3_ngilt = [], [], []
iter1_gilmp, iter2_gilmp, iter3_gilmp = [], [], []
iter1_gilt, iter2_gilt, iter3_gilt = [], [], []

def store_lists(file_, list1, list2, list3):
	for ind, temp in enumerate(file_.readlines()):
		if ind < 9:
			list_ = list1
		elif ind >= 9 and ind < 18:
			list_ = list2
		else:
			list_ = list3
		for num in temp.strip().split(' '):
			list_.append(float(num))


store_lists(m_nogil, iter1_ngilmp, iter2_ngilmp, iter3_ngilmp)
store_lists(t_nogil, iter1_ngilt, iter2_ngilt, iter3_ngilt)
store_lists(m_gil, iter1_gilmp, iter2_gilmp, iter3_gilmp)
store_lists(t_gil, iter1_gilt, iter2_gilt, iter3_gilt)

mean_ngilmp = np.mean([iter1_ngilmp, iter2_ngilmp, iter3_ngilmp], axis=0)
std_ngilmp = np.std([iter1_ngilmp, iter2_ngilmp, iter3_ngilmp], axis=0)
mean_ngilt = np.mean([iter1_ngilt, iter2_ngilt, iter3_ngilt], axis=0)
std_ngilt = np.std([iter1_ngilt, iter2_ngilt, iter3_ngilt], axis=0)
mean_gilmp = np.mean([iter1_gilmp, iter2_gilmp, iter3_gilmp], axis=0)
std_gilmp = np.std([iter1_gilmp, iter2_gilmp, iter3_gilmp], axis=0)
mean_gilt = np.mean([iter1_gilt, iter2_gilt, iter3_gilt], axis=0)
std_gilt = np.std([iter1_gilt, iter2_gilt, iter3_gilt], axis=0)

ngilmp_core1, ngilmp_core2, ngilmp_core3 = (mean_ngilmp[:12],
    mean_ngilmp[12:24], mean_ngilmp[24:])
ngilt_core1, ngilt_core2, ngilt_core3 = (mean_ngilt[:12],
	mean_ngilt[12:24], mean_ngilt[24:])
gilmp_core1, gilmp_core2, gilmp_core3 = (mean_gilmp[:12],
	mean_gilmp[12:24], mean_gilmp[24:])
gilt_core1, gilt_core2, gilt_core3 = (mean_gilt[:12],
	mean_gilt[12:24], mean_gilt[24:])

sngilmp_core1, sngilmp_core2, sngilmp_core3 = (std_ngilmp[:12],
    std_ngilmp[12:24], std_ngilmp[24:])
sngilt_core1, sngilt_core2, sngilt_core3 = (std_ngilt[:12],
	std_ngilt[12:24], std_ngilt[24:])
sgilmp_core1, sgilmp_core2, sgilmp_core3 = (std_gilmp[:12],
	std_gilmp[12:24], std_gilmp[24:])
sgilt_core1, sgilt_core2, sgilt_core3 = (std_gilt[:12],
	std_gilt[12:24], std_gilt[24:])


# Plotting bar plots in matplotlib
fig, (axis1, axis2, axis3) = pl.subplots(3, 1, sharex=True)
ind = np.arange(12)
width = 0.35

axis1.set_title("n_jobs=1, MULTIPROCESS")
bar1m = axis1.bar(ind, ngilmp_core1, width, color="r", yerr=sngilmp_core1)
bar2m = axis1.bar(ind + width, gilmp_core1, width, color="y", yerr=sgilmp_core1)
axis1.set_ylabel("Time")
axis1.set_xticks(ind + width)
axis1.set_xticklabels(('1, 5', '1, 10', '1, 50', '1, 100',
	                  '2, 5', '2, 10', '2, 50', '2, 100',
	                  '5, 5', '5, 10', '5, 50', '5, 100'))
axis1.legend((bar1m[0], bar2m[0]), ("In this branch", "In master"), loc="upper left")

axis2.set_title("n_jobs=2")
bar1m = axis2.bar(ind, ngilmp_core2, width, color="r", yerr=sngilmp_core2)
bar2m = axis2.bar(ind + width, gilmp_core2, width, color="y", yerr=sgilmp_core2)
axis2.set_ylabel("Time")
axis2.legend((bar1m[0], bar2m[0]), ("In this branch", "In master"), loc="upper left")

axis3.set_title("n_jobs=4")
bar1m = axis3.bar(ind, ngilmp_core3, width, color="r", yerr=sngilmp_core3)
bar2m = axis3.bar(ind + width, gilmp_core3, width, color="y", yerr=sgilmp_core3)
axis3.set_xlabel("no. of L1 ratio | no. of alpha")
axis3.set_ylabel("Time")
axis3.legend((bar1m[0], bar2m[0]), ("In this branch", "In master"), loc="upper left")

pl.savefig("Benching with multiprocessing.png")

axis1.set_title("n_jobs=1, THREADING")
bar1m = axis1.bar(ind, ngilt_core1, width, color="r", yerr=sngilt_core1)
bar2m = axis1.bar(ind + width, gilt_core1, width, color="y", yerr=sgilt_core1)
axis1.set_ylabel("Time")
axis1.set_xticks(ind + width)
axis1.set_xticklabels(('1, 5', '1, 10', '1, 50', '1, 100',
	                  '2, 5', '2, 10', '2, 50', '2, 100',
	                  '5, 5', '5, 10', '5, 50', '5, 100'))
axis1.legend((bar1m[0], bar2m[0]), ("In this branch", "In master"), loc="upper left")

axis2.set_title("n_jobs=2")
bar1m = axis2.bar(ind, ngilt_core2, width, color="r", yerr=sngilt_core2)
bar2m = axis2.bar(ind + width, gilt_core2, width, color="y", yerr=sgilt_core2)
axis2.set_ylabel("Time")
axis2.legend((bar1m[0], bar2m[0]), ("In this branch", "In master"), loc="upper left")

axis3.set_title("n_jobs=4")
bar1m = axis3.bar(ind, ngilt_core3, width, color="r", yerr=sngilt_core3)
bar2m = axis3.bar(ind + width, gilt_core3, width, color="y", yerr=sgilt_core3)
axis3.set_xlabel("no. of L1 ratio | no. of alpha")
axis3.set_ylabel("Time")
axis3.legend((bar1m[0], bar2m[0]), ("In this branch", "In master"), loc="upper left")
pl.savefig("Benching with threading.png")


m_nogil.close()
t_nogil.close()
m_gil.close()
t_gil.close()