from math import *
from numpy import *


def loadDataSet():
    dataMat = []
    labelMat = []
    with open('testSet.txt') as fr:
        for line in fr:
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])   # 这个1方便bias
            labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


if __name__ == '__main__':
    dataArr, labelMat = loadDataSet()
    result = gradAscent(dataArr, labelMat)
    print(result)
