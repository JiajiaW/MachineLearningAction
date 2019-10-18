import numpy as np
import operator


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    K-近邻算法
    :param inX: 输入向量
    :param dataSet: 输入的训练样本集
    :param labels: 标签
    :param k: 选择的最近邻居的数目
    :return:
    """
    # 1. 求欧式距离
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet   # 将输入向量重复数据集长度个，对应相减
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5

    # 2. 选择距离最小的k个点，并得到标签，统计每个标签的个数
    sortedDistIndicies = distances.argsort()  # 得到排序后索引值 如[2,3,1,0]
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 3. 从字典中得到标签数量最多的标签
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


if __name__ == '__main__':
    group, labels = createDataSet()
    result = classify0([0, 0], group, labels, 3)
    print(result)
