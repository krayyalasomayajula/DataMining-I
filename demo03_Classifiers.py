'''
Created on Sept 09, 2016

@author: kalyan
'''
# Import datasets, classifiers and performance metrics
import numpy as np
import cv2

from sklearn import neighbors, metrics

import pandas as pd

import pdb

irisDF = pd.io.parsers.read_csv(
    './iris_data.csv',
     header=None, delimiter=",",
     usecols=[1,2,3,4],
     skiprows=1
    )

irisLF = pd.io.parsers.read_csv(
    './iris_labels.csv',
     header=None, delimiter=",",
     usecols=[1],
     skiprows=1
    )

#Pickrows froms 101 to 500
'''
irisDF = pd.io.parsers.read_csv(
    './iris_data.csv',
     header=None, delimiter=",",
     usecols=[1,2,3,4],
     skiprows=100,
     nrows=500
    )
'''
data = irisDF.as_matrix()
labels = (irisLF.as_matrix())[:,0]
n_samples = len(labels)

assert(data.shape[0] == len(labels))

#pdb.set_trace()

#Split the data into training and test set in 70-30 split
split = 0.7
n_trainSamples = int(n_samples*split) 
trainData   = data[:n_trainSamples,:]
trainLabels = labels[:n_trainSamples]

testData    = data[n_trainSamples:,:]
expectedLabels  = labels[n_trainSamples:]

# k-NearestNeighbour Classifier instance
n_neighbors = 10
kNNClassifier = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
# trains the model
kNNClassifier.fit(trainData, trainLabels) 

predictedLabels = kNNClassifier.predict(testData)

#Display classifier results
print("Classification report for classifier %s:\n%s\n"
      % ('k-NearestNeighbour', metrics.classification_report(expectedLabels, predictedLabels)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expectedLabels, predictedLabels))

print('Done.')

