#! /usr/bin/env python
from __future__ import division, print_function
import cv2
import argparse
import numpy as np
#import matplotlib.pyplot as plt
from scipy import signal
import sys

#Savitzky Method Adapted from Miguel Carvajal - https://gist.github.com/krvajal/1ca6adc7c8ed50f5315fee687d57c3eb
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def calculatepeaks(vet, a, b):
        saida = []
        i = 0
        while (i < len(vet)-1):
            j = i
            while(vet[j] <= vet[j+1] and j<(len(vet)-3)):
                j+=1
                k=j
            while(vet[j] >= vet[j+1] and j<(len(vet)-3)):
                j+=1
            minimo = min(abs(j-k),abs(k-i))
            #verifica se o pico tem largura minima a
            if (minimo>a):
                #verifica se o pico tem altura minima de valor b
                if(((abs(vet[k+minimo]-vet[k])+abs(vet[k-minimo]-vet[k]))/2)>b):
                        saida.append(k)
            i = j+1
        return saida

def desenhalinhas(mat, vet):
    for i in range(2,mat.shape[0]-1):
        if(i in vet):
            mat[i-2:i+1][:]=[0,0,255]
    return mat

def rotatec(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=255)

def rotate(img, angle):
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    return (cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_CONSTANT, borderValue=255))

def findrotation(im):
    resultado = np.zeros(len(range(-10,10)))
    for ang in range(-10,10):
            img = rotate(im, ang)
            i = img.shape[0]
            j = img.shape[1]
            vetrot = np.zeros(i)
            for ii in range(0, i):
                    vetrot[ii] = j - (np.sum(img[ii])/255)
            vetrot = savitzky_golay(vetrot, 71, 2, 0)
            resultado[ang] = np.amax(vetrot) 
    return resultado

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1], 0)
    if img is None:
        exit()
    img2 = cv2.GaussianBlur(img,(5,5),0)
    _, img_p = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    seg = img_p[:,0:500]
    i = seg.shape[0]
    j = seg.shape[1]
    vetor = np.zeros(i)
    for ii in range(0, i):
            vetor[ii] = j - (np.sum(seg[ii])/255)
    vetor = savitzky_golay(vetor, 53, 2, 0)
    
		#calcula valores de soma dos pixels por linha para rotacoes de -10 a +10 graus
    rota = findrotation(img_p)
    esqu = (rota[9]/rota[10])*100
    dire = (rota[11]/rota[10])*100

		#verifica se a porcentagem de variacao entre a rotacao 0 e -1 e +1 grau difere em valor da soma dos pixels pretos por linha em mais de 20%
    if (abs(esqu-100)>20 or abs(dire-100)>20):
            img = rotate(img, -(np.argmax(rota)-10))
            img2 = cv2.GaussianBlur(img,(5,5),0)
            _, img_p = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            seg = img_p[:,0:500]
            i = seg.shape[0]
            j = seg.shape[1]
            vetor = np.zeros(i)
            for ii in range(0, i):
                    vetor[ii] = j - (np.sum(seg[ii])/255)
            vetor = savitzky_golay(vetor, 65, 2, 0) 
    
		#calcula picos, que sao indicativos de linhas de texto, remove os repetidos
    saida = np.unique(calculatepeaks(vetor, 16, 10))
    im = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    print(len(saida))
    img = desenhalinhas(im, saida)
    cv2.imwrite("s_"+sys.argv[1],im)
