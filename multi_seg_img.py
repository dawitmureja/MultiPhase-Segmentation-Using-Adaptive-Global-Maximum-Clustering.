import numpy as np
from AGMC import AGMC
import ipdb
from scipy.misc import imread,imshow,imsave
from plot_hist import plotHistogram
eps = 2.2204e-16
def rescale(x,im,iM,om,oM):
    if (len(im) == 0):
        im = np.amin(x)
    if (len(iM) == 0):
        iM = np.amax(x)
    x[x<im],x[x>iM] = im,iM
    if iM-im < eps:
        y = x
    else:
        y = ((oM-om) * (x-im)/float(iM-im)) + om
    return y
#x = np.arange(12).reshape(2,2,3)
#print(x)
#y = rescale(x,[],[],0,1)
#print(y)
#ipdb.set_trace()
img = imread('spine.png', flatten = True)
#ipdb.set_trace()
#imshow(img)
#print(img)
#print(img[0])
sgm = 20
omg = 0.5
index = 0

def multiphaseSegmentation(img3D, sgm, omg):
    hist = np.histogram(img3D, bins = 256)
    #hist = np.histogram(img3D, bins = 256)
    bin0 = hist[1][:-1]
    h0 = hist[0]
    img_scale = rescale(img, [],[],0,1)
    hist_scale = np.histogram(img, bins = np.arange(1,258))
    counts, binr = hist_scale[0],hist_scale[1][:-1]
    h = counts.astype(float) / np.sum(counts)
    Intv_str,Intv_end,h0 = AGMC(h0,sgm,omg)
    N = len(Intv_str)
    #plotHistogram(Intv_str, Intv_end, binr, h)
    plotHistogram(Intv_str, Intv_end, bin0, h0)
    #ipdb.set_trace()
    c = np.zeros(Intv_str.shape)
    for i in range(N):
        Intv = np.arange(Intv_str[i]-1, Intv_end[i])
        c[i] = np.sum(binr[Intv] * h0[Intv])/ np.sum(h0[Intv] + eps)
    c = np.sort(c)
    #u,init_phi = img_scale, np.zeros(img_scale.shape)
    u,init_phi = img , np.zeros(img_scale.shape)
    #Min_org, Max_org = np.amin(u), np.amax(u)
    for k in range(1,N+1):
        if k == 1:
            tmp = (c[0] + c[1])/2
            init_phi[u <= tmp] = k
        elif k == N:
            tmp = (c[k-2] + c[k-1])/2
            init_phi[tmp < u] = k
        else:
            tmp1,tmp2 = (c[k-2] + c[k-1])/2, (c[k-1] + c[k])/2
            tmp3 = np.all([u > tmp1, u <= tmp2],axis = 0)
            init_phi[tmp3] = k
    phi_AGMC = init_phi
    max_iter, tol = 10, 1e-10
    phi = init_phi
    oldphi= np.ones(img_scale.shape)
    for iter in range(max_iter):
        for k in range(1, N+1):
            c[k-1] = np.mean(u[phi == k])
        for k in range(2, N):
            tmp = np.all([phi == k, (c[k-2] + c[k-1])/2 >= u], axis = 0)
            phi[tmp] = k-1
        tmp = np.all([phi == N, (c[N-2] + c[N-1])/2 >  u], axis = 0)
        phi[tmp] = N-1
        err = np.sum(np.sqrt(np.square(phi-oldphi)/ phi.size))
        if err < tol:
            break
        oldphi = phi
    phi_AGMC = phi
    #final = (rescale(phi_AGMC,[],[],0,1))
    #imshow(final)
    #imshow(phi_AGMC)
    imsave('mult_result.png',phi_AGMC)
    return phi_AGMC

multiphaseSegmentation(img,sgm, omg)




