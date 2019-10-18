from math import log
import operator


def calcShannonEnt(dataSet):
    """
    计算给定数据集的香农熵
    :param dataSet:
    :return:
    """
    # 1. 计算每种标签的个数和总个数，为后续的概率计算做准备
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]

        # 这个可以使用 labelCounts.get(currentLabel, 0) + 1  代替
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    # 2. 计算概率，并最后计算熵
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt


def splitDataSet(dataSet, axis, value):
    """
    按照给定特征划分数据集
    :param dataSet: 待划分的数据集
    :param axis: 划分数据集的特征
    :param value: 指定的特征值
    :return:
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """
    根据信息增益获得最优的特征划分数据集，决策树中最关键的算法
    :param dataSet:
    :return:
    """
    numFeatures = len(dataSet[0]) - 1      # the last column is used for the labels 特征个数
    baseEntropy = calcShannonEnt(dataSet)  # 计算数据集的熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):        # iterate over all the features 遍历所有特征
        # 得到唯一的分类标签
        featList = [example[i] for example in dataSet]  # create a list of all the examples of this feature
        uniqueVals = set(featList)       # get a set of unique values 得到当前特征所有可能的值
        newEntropy = 0.0
        # 遍历特征所有可能的值，求出以每个值进行分类时子集的熵，将概率乘以熵得到最后的新熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 计算最好的信息增益
        infoGain = baseEntropy - newEntropy     # calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       # compare this to the best gain so far
            bestInfoGain = infoGain         # if better than current best, set to best
            bestFeature = i
    return bestFeature                      # returns an integer 返回信息增益最大的特征值所在的索引


def majorityCnt(classList):
    """
    在最后要确定数据的标签，并且数据中不止一个标签时，使用投票法决定最后的标签
    :param classList:
    :return:
    """
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    """
    决策树实现的主要函数，递归的方法
    1. 两个终止条件：子数据集中只剩一种标签；数据没有更多特征能够使用
    2. 选择要使用哪一个特征去划分 信息增益
    3. 对选择的特征的所有值进行遍历，获取根据特征值划分的子数据集，并对子数据集进行递归
    :param dataSet:
    :param labels:
    :return:
    """
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1:  # stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 选择要使用哪一个特征去划分
    bestFeatLabel = labels[bestFeat]  # 获取特征索引对应的真实属性名称
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])  # 删除已经使用过的标签
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       # copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels


if __name__ == '__main__':
    myDat, labels = createDataSet()
    print(myDat)
    result = calcShannonEnt(myDat)
    print(result)

    # myDat[0][-1] = 'maybe'   # 熵越高，混合的数据也越多
    # print(myDat)
    # result = calcShannonEnt(myDat)
    # print(result)

    # result = splitDataSet(myDat, 0, 1)
    # print(result)

    # result = chooseBestFeatureToSplit(myDat)
    # print(result)

    result = createTree(myDat, labels)
    print(result)



