import numpy as np
import matplotlib.pyplot as plt

def choose_color(k):
    if k % 5 == 0:
        color = 'r-'
    elif k % 5 == 1:
        color = 'b-'
    elif k % 5 == 2:
        color = 'g-'
    elif k % 5 == 3:
        color = 'c-'
    elif k %5 == 4:
        color = 'm-'
    return color


def plotHistogram(Intv_str, Intv_end, bin0, h0_):
    N = len(Intv_str)
    plt.figure(1)
    plt.plot(bin0,h0_,'k-', lw = 1.5)
    for k in range(N-1):
        color = choose_color(k)
        plt.plot(bin0[Intv_str[k]-1: Intv_end[k] + 1], h0_[Intv_str[k]-1: Intv_end[k]+1], color, lw = 2.5)
    color = choose_color(N-1)
    plt.plot(bin0[Intv_str[N-1]-1:Intv_end[N-1]+1], h0_[Intv_str[N-1]-1:Intv_end[N-1]+1], color, lw = 2.5)
    plt.title("# of clusters: %d" % N)
    plt.show()


