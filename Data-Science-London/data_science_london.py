from numpy import *
import csv

def loadData(csvname):
    l = []
    with open(csvname) as f:
        lines = csv.reader(f)
        for line in lines:
            l.append(line)
    l = array(l)            #read the data and transfer into the array type
    return string_to_int(l)

def string_to_int(array):
    array = mat(array)
    m,n = shape(array)      #get the dimention of the array
    newarray = zeros((m,n))
    for i in range(m):
        for j in range(n):
            newarray[i,j] = float(array[i,j])
    return newarray

def saveResult(result,csvName):
    with open(csvName,"w",newline='') as f:
        writer = csv.writer(f)
        tmp = []
        tmp.append('Id')
        tmp.append('Solution')
        writer.writerow(tmp)
        for i in range(len(result)):
            tmp = []
            tmp.append((i+1))
            tmp.append(result[i])
            writer.writerow(tmp)

#调用knn算法包
from sklearn.neighbors import KNeighborsClassifier
def knnClassify(trainData,trainLabel,testData):
    knnClf = KNeighborsClassifier(10)
    knnClf.fit(trainData,ravel(trainLabel))     #每一行作为一个数据
    testLabel = knnClf.predict(testData)
    saveResult(testLabel,'sklearn_knn_Result.csv')
    return testLabel

#调用logistics回归算法
from sklearn.linear_model import LogisticRegression
def logisticRegressionClf(trainData,trainLabel,testData):
    lrClf=LogisticRegression()
    lrClf.fit(trainData,ravel(trainLabel))
    testLabel=lrClf.predict(testData)
    saveResult(testLabel,'sklearn_lr_Result.csv')
    return testLabel

#调用scikit的SVM算法包
from sklearn import svm
def svcClassify(trainData,trainLabel,testData):
    svcClf=svm.SVC(C=5) #default:C=1.0,kernel = 'rbf'. you can try kernel:‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    svcClf.fit(trainData,ravel(trainLabel))
    testLabel=svcClf.predict(testData)
    saveResult(testLabel,'sklearn_SVC_C=5.0_Result.csv')
    return testLabel

#调用scikit的朴素贝叶斯算法包,GaussianNB和MultinomialNB
from sklearn.naive_bayes import GaussianNB      #nb for 高斯分布的数据
def GaussianNBClassify(trainData,trainLabel,testData):
    nbClf=GaussianNB()
    nbClf.fit(trainData,ravel(trainLabel))
    testLabel=nbClf.predict(testData)
    saveResult(testLabel,'sklearn_GaussianNB_Result.csv')
    return testLabel

def DataScienceLondon():
    trainData=loadData('train.csv')
    trainLabel=loadData('trainLabels.csv')
    testData=loadData('test.csv')
    #使用不同算法
    knnClassify(trainData,trainLabel,testData)     #
    svcClassify(trainData,trainLabel,testData)     #
    logisticRegressionClf(trainData,trainLabel,testData)
    GaussianNBClassify(trainData,trainLabel,testData)  #效果最好

if __name__ == '__main__':
    DataScienceLondon()