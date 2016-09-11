'''
Created on Sept 09, 2016

@author: kalyan
'''
from math import sqrt, pow, sin, cos
import numpy as np
import random

import pdb

c=15.0
b=5.0
a=2.0
stepSz = 0.1
theta = 60.0/360.0
cords = []

i=0
for x in np.arange(-c,c,stepSz).tolist():
    for y in np.arange(-b,b,stepSz).tolist():
        zmax = a*sqrt(min(1.0,max(0.0, 1.0 - pow(x,2)/pow(c,2) - pow(y,2)/pow(b,2))))
        zmin = -a*sqrt(min(1.0,max(0.0, 1.0 - pow(x,2)/pow(c,2) - pow(y,2)/pow(b,2))))
        for z in np.arange(zmin,zmax,stepSz).tolist():
            cords.append([x,y,z])
random.shuffle(cords)
cords = np.asarray(cords)
cords = cords[0:2000,:]

#cords = np.dot(cords, np.array([[cos(theta), -sin(theta), 0],[sin(theta), cos(theta), 0],[0, 0, 1]]))
#cords = np.dot(cords, np.array([[cos(theta), 0, sin(theta)],[0, 1, 0],[-sin(theta), 0, cos(theta)]]))
#pdb.set_trace()
#print len(cords)
'''
Plot using Matplotlib 
'''
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot3D(xCords,col):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(xCords[:,0], xCords[:,1], xCords[:,2], color=col,
            label='Standardized [$N  (\mu=0, \; \sigma=1)$]', alpha=0.3)

    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_zlabel('Z-Axis')
    plt.grid()
    plt.tight_layout()

plot3D(cords,'red')
#plt.show()

from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2,whiten=True)
Y_sklearn = sklearn_pca.fit_transform(cords)
XX = sklearn_pca.inverse_transform(Y_sklearn)

plot3D(Y_sklearn,'green')
plt.show()
