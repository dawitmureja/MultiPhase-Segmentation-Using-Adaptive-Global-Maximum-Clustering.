import numpy as np
import ipdb
import copy
from scipy.misc import imread
import matplotlib.pyplot as plt
from kmeans import fast_kmeans
#from plot_hist import plotHistogram

def AGMC(h,sgm,omg):
    confirm_ind,del_ind,live_ind,Intv_str,Intv_end = [],[],[],[],[]
    h0 = h.copy()
    outer_Max_iter = 10
    for outer_iter in range(outer_Max_iter):
        if outer_iter == 0:
            h_ind = np.arange(1,257)

        in_Max_iter = 10
        for in_iter in range(in_Max_iter+1):
            pre_h_ind = h_ind
            mu,mask = fast_kmeans(h,2,h_ind)
            ind1,ind2 = (np.where(mask == 1)[0],np.where(mask == 2)[0])

            tmp_ind1, tmp_ind2 = (np.argmax(h[ind1]), np.argmax(h[ind2]))
            Max1, Max2 = (ind1[tmp_ind1], ind2[tmp_ind2])
            if (Max1 > Max2):
                Max2, Max1 = (ind1[tmp_ind1], ind2[tmp_ind2])
            if (np.amax(h[ind1]) > np.amax(h[ind2])):
                Cluster, Mu, Max = (1,  mu[0], Max1)
            else:
                Cluster, Mu, Max = (2,  mu[1], Max2)
            if (Cluster == 1):
                dind1 = ind1[1:] - ind1[0:-1]
                if (np.sum(dind1 != 1) != 0):
                    Nb, tmp_area = 1,[]
                    for q in range(len(ind1)-1):
                        tmp_area.append(Nb)
                        if (dind1[q] != 1):
                            Nb += 1
                    tmp_area.append(Nb)
                    tmp_area = np.array(tmp_area)
                    for p in range(1,Nb + 1):
                        if (np.sum(ind1[tmp_area == p] == Max1) != 0):
                            h_ind = ind1[tmp_area == p] + 1
                        else:
                           h[ind1[tmp_area == p]] = 0
                else:
                    #shifing the index so that it will start from 1
                    h_ind = ind1 + 1
                h[ind2] = 0
            else:
                dind2 = ind2[1:] - ind2[0:-1]
                if (np.sum(dind2 != 1) != 0):
                    Nb, tmp_area = 1,[]
                    for q in range(len(ind2)-1):
                        tmp_area.append(Nb)
                        if (dind2[q] != 1):
                             Nb += 1
                    tmp_area.append(Nb)
                    tmp_area = np.array(tmp_area)
                    for p in range(1,Nb+1):
                        if (np.sum(ind2[tmp_area == p] == Max2) != 0):
                             h_ind = ind2[tmp_area == p] + 1
                        else:
                             h[ind2[tmp_area==p]] = 0
                else:
                    h_ind = ind2 + 1
                h[ind1] = 0
        
            if np.absolute(Max1 - Max2) < sgm:
                #print("F this shit")
                del_ind = []
                if (len(live_ind) == 0):
                    i_ind = np.arange(1,257)
                else:
                    i_ind = np.sort(live_ind)
                    
                for i in range(i_ind[0], i_ind[-1] + 1):
                    if (np.sum(i == pre_h_ind) == 0):
                        h[i-1] = h0[i-1]
                    elif (np.sum(i == pre_h_ind) != 0):
                        del_ind = np.concatenate((del_ind, [i]), axis = 0)
                        h[i-1] = 0
                #print(h)
                confirm_ind = np.concatenate((del_ind, confirm_ind), axis = 0)
                break
        h_ind = []
        #print(del_ind)
        for i in range(1,257):
            if (np.sum(i == confirm_ind) == 0):
                h_ind = np.concatenate((h_ind, [i]), axis = 0)
        live_ind = np.array(h_ind).astype(int)
        #print(h_ind)
        if (len(live_ind) == 0) or (np.amax(h[live_ind-1]) < (omg * np.mean(h0))):
            Intv_str = np.concatenate((Intv_str,[pre_h_ind[0]]), axis = 0)
            Intv_end = np.concatenate((Intv_end,[pre_h_ind[-1]]), axis = 0)
            break
        Intv_str = np.concatenate((Intv_str,[del_ind[0]]), axis = 0)
        Intv_end = np.concatenate((Intv_end,[del_ind[-1]]), axis = 0)
        if len(live_ind) == 0:
            break
    return np.sort(Intv_str).astype(int), np.sort(Intv_end).astype(int),h0











