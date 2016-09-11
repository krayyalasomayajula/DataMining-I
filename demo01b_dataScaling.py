'''
Created on Sept 09, 2016

@author: kalyan
'''
import pandas as pd
import numpy as np

df = pd.io.parsers.read_csv(
    './wine.data',
     header=None,
     usecols=[0,1,2,3]
    )

df.columns=['Class label', 'Alcohol', 'Malic acid','Magnesium']

df.head()

#print df.values
'''
Pre-processing module for data normalization
'''
from sklearn import preprocessing

std_scale = preprocessing.StandardScaler().fit(df[['Alcohol', 'Malic acid','Magnesium']])
df_std = std_scale.transform(df[['Alcohol', 'Malic acid','Magnesium']])

minmax_scale = preprocessing.MinMaxScaler().fit(df[['Alcohol', 'Malic acid','Magnesium']])
df_minmax = minmax_scale.transform(df[['Alcohol', 'Malic acid','Magnesium']])

print('Mean after standardization:\nAlcohol={:.2f}, Malic acid={:.2f}, Magnesium={:.2f}'
      .format(df_std[:,0].mean(), df_std[:,1].mean(), df_std[:,2].mean()))
print('\nStandard deviation after standardization:\nAlcohol={:.2f}, Malic acid={:.2f}, Magnesium={:.2f}'
      .format(df_std[:,0].std(), df_std[:,1].std(), df_std[:,2].std()))


print('Min-value after min-max scaling:\nAlcohol={:.2f}, Malic acid={:.2f}, Magnesium={:.2f}'
      .format(df_minmax[:,0].min(), df_minmax[:,1].min(), df_minmax[:,2].min()))
print('\nMax-value after min-max scaling:\nAlcohol={:.2f}, Malic acid={:.2f}, Magnesium={:.2f}'
      .format(df_minmax[:,0].max(), df_minmax[:,1].max(), df_minmax[:,2].max()))

'''
Plot using Matplotlib 
'''
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot3D():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(df['Alcohol'], df['Malic acid'], df['Magnesium'],
            color='green', label='input scale', alpha=0.5)

    ax.scatter(df_std[:,0], df_std[:,1], df_std[:,2], color='red',
            label='Standardized [$N  (\mu=0, \; \sigma=1)$]', alpha=0.3)

    ax.scatter(df_minmax[:,0], df_minmax[:,1], df_minmax[:,2],
            color='blue', label='min-max scaled [min=0, max=1]', alpha=0.3)

    plt.title('Alcohol, Malic Acid and Magnesium content of the wine dataset')
    ax.set_xlabel('Alcohol')
    ax.set_ylabel('Malic Acid')
    ax.set_zlabel('Magnesium')
    plt.legend(loc='upper left')
    plt.grid()

    plt.tight_layout()

plot3D()
plt.show()

