from sklearn.cluster import KMeans
from numpy import genfromtxt
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

s = zeroes.np((56, 9))
for i in range( 1, 57):
    fname = floc + i + '.csv'
    data = genfromtxt( fname, delimiter=',')
    for j in range(2, 11):
        kmeans = KMeans(n_clusters= j , random_state=0).fit( data[:,0:-1])
        s[i-1, j-2] = silhouette_score(data[:,0:-1], kmeans.labels_, metric='euclidean');
        plt.boxplot(data)
