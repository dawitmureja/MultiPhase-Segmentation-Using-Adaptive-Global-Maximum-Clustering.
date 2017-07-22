import numpy as np
import ipdb
import copy
from scipy.misc import imshow, imread
eps = 2.2204e-16

#img = imread('sample.jpg', flatten = True)
#hist = np.histogram(img,bins = np.arange(1,257))
#print(hist[0])
def fast_kmeans(h,k,h_ind):
    ind = h_ind.astype(int)
    mu = [ind[0], ind[-1]]
    iter = 0
    max_iter = 3
    hc = np.zeros((np.amax(ind),1))
    while (iter < max_iter):
        iter += 1
        oldmu = [x+y for x,y in zip(mu,[0,0])]
        for i in range(len(ind)):
            c = np.absolute(ind[i] - mu)
            hc[ind[i]-1] = (np.where(c == c.min())[0][0]) + 1
        #hc += 1
            #hc = np.concatenate((hc, np.where(c == c.min())[0]), axis = 0)
        for j in range(1,k+1):
            a = np.where(hc == j)[0]
            if (np.sum(h[a]) == 0):
                mu[j-1] = np.mean(a)
            else:
                mu[j-1] = np.sum((a+1) * h[a]) / np.sum(h[a] + eps)
        mask = hc.T[0]
        if (mu == oldmu):
            mask = hc.T[0]
            break
    return mu, mask
#h = np.random.randint(10, size = (1,10))[0]
#h_ind = np.arange(1,11)
#k = 2
#a,b = fast_kmeans(h,k,h_ind)
#print(h)
#print(a)
#print(b)
#ipdb.set_trace()
