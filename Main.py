# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 10:44:51 2016

@author: gazalchawla
"""

import numpy as np
import operator
import random as rd
import datetime


# method computes the euclidean distance
def euclideanDistance(instance1, instance2):
    distance = 0
    distance = np.linalg.norm(instance1-instance2)
    return distance


# method finds the k nearest neighbours for a test instance
def getNeighbours(testInstance, trainSet, labels, k):
    distances = []
    for row, l in zip(trainSet, labels):
        dist = euclideanDistance(testInstance, row)
        distances.append((l, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbours = []
    for x in range(k):
        neighbours.append(distances[x][0])
    return neighbours


# method predicts label for one test instance
def predictLabel(neighbours):
    classVotes = {}
    for x in range(len(neighbours)):
        label = neighbours[x][-1]
        if label in classVotes:
            classVotes[label] += 1
        else:
            classVotes[label] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(0), reverse=True)
    return sortedVotes[0][0]


# method compared predicted labels to the original labels to compute accuracy
def getAccuracy(testSet, predictedSet):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][0] == predictedSet[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


# method returns predicted labels as a numpy array
def testKnn(trainX, trainY, testX, k):
    predictedSet = []
    for row in testX:
        neighbours = getNeighbours(row, trainX, trainY, k)
        result = predictLabel(neighbours)
        predictedSet.append(result)
    return (np.asarray(predictedSet))


# method generates a random index
def randomIndex(array, i):
    index = rd.randint(1, i)
    while array[index]:
        index = rd.randint(1, i)
    if not array[index]:
        array[index] = True
    return index


# method to condense data
def condenseData(trainX, trainY):
    count = len(trainX)
    array = [False for i in range(0, count)]
    boolArray = np.array(array)
    boolArray[0] = True
    subsetX = []
    subsetX.append(trainX[0])
    subsetY = []
    subsetY.append(trainY[0])
    condensedIdx = []
    flagArray = np.all(boolArray == True)
    while not flagArray:
        currentTest = []
        i = randomIndex(boolArray, count-1)
        currentTest.append(trainX[i])
        testY = testKnn(subsetX, subsetY, currentTest, 1)
        if(testY[0] != trainY[i]):
            subsetX.append(trainX[i])
            subsetY.append(trainY[i])
            boolArray[i] = True
            condensedIdx.append(i)
        flagArray = np.all(boolArray == True)
    print(repr(len(subsetY)))
    return (np.asarray(condensedIdx))


def main():
    
    # uncomment to execute condensed knn    
    '''print('Main Starts...')    
    xData = np.loadtxt('letter-recognition.data', dtype='int', delimiter=',', usecols=range(1, 17))
    trainX = xData[0:5000]
    testX = xData[15000:20000]
    yData = np.loadtxt('letter-recognition.data', dtype='string', delimiter=',', usecols=[0])
    trainY = yData[0:5000]
    testYActual = yData[15000:20000]
    k = 9
    testY = np.array([])
    condensedIdx = np.array([])
    startTime = datetime.datetime.now()
    condensedIdx = condenseData(trainX, trainY)
    
    print('Time to find the condensed subset: ')
    print(datetime.datetime.now()-startTime)
    updatedTrainX = np.array([])
    updatedTrainX = trainX[condensedIdx,:]
    updatedTrainY = np.array([])
    for i in condensedIdx:
        updatedTrainY = np.append(updatedTrainY, trainY[[i]]) 
    print('Started at :')
    m = datetime.datetime.now()
    print(startTime)
    print('Finding k nearest neighbours...')

    testY = testKnn(updatedTrainX, updatedTrainY, testX, k)
    print('Neighbours found...')
    endTime = datetime.datetime.now()
    print('Ended at: ')
    print(endTime)
    print('Time to work on knn: ')
    print(endTime-m)
    print('Total time elapsed: ')
    print(endTime-startTime)
    accuracy = getAccuracy(testYActual, testY)
    print('Accuracy: ' + repr(accuracy) + '%')'''

    # uncomment to execute knn
    '''print('Main Starts...')    
    xData = np.loadtxt('letter-recognition.data', dtype='int', delimiter=',', usecols=range(1, 17))
    trainX = xData[0:15000]
    testX = xData[15000:20000]
    yData = np.loadtxt('letter-recognition.data', dtype='string', delimiter=',', usecols=[0])
    trainY = yData[0:15000]
    testYActual = yData[15000:20000]
    
    # indexArray generates random indices, attribute size = N gives an array of N random indices
    # change this N value for subsampling
    indexArray = np.random.randint(15000, size=15000)
    trainXRandom = trainX[indexArray,:]
    trainYRandom = np.array([])
    for i in indexArray:
        trainYRandom = np.append(trainYRandom, trainY[[i]])
    k = 9
    testY = np.array([])
    startTime = datetime.datetime.now()
    print('Started at:')
    print(startTime)
    print('Finding k nearest neighbours...')

    testY = testKnn(trainXRandom, trainYRandom, testX, k)
    print('Neighbours found...')
    endTime = datetime.datetime.now()
    print('Ended at: ')
    print(endTime)
    print('Time to work on knn: ')
    print(endTime-startTime)
    accuracy = getAccuracy(testYActual, testY)
    print('Accuracy: ' + repr(accuracy) + '%')'''


main()
