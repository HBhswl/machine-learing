from numpy import *
import csv

def nomalizing(array):
    m,n = shape(array)
    for i in range(m):
        for j in range(n):
            if array[i,j] != 0:
                array[i,j] = 1
    return array

def string_to_int(array):
    array = mat(array)
    m,n = shape(array)
    newarray = zeros((m,n))
    for i in range(m):
        for j in range(n):
            newarray[i,j] = int(array[i,j])
    return newarray


def load_train_data():
    l = []
    with open('train.csv') as f:
         lines=csv.reader(f)
         for line in lines:
             l.append(line) #42001*785  #store the train data
    l.remove(l[0])      #remove the first line which contains the pixels
    l = array(l)        #transfer to the array type
    label = l[:,0]  #read the label of every data
    data = l[:,1:]  #read the pixel data of every data
    return nomalizing(string_to_int(data)), string_to_int(label)    #binaryzation and transfer string to int

def loadTestData():
    l=[]
    with open('test.csv') as file:
         lines=csv.reader(file)
         for line in lines:
             l.append(line)#28001*784
    l.remove(l[0])
    data=array(l)
    return nomalizing(string_to_int(data))  #  data 28000*784
    #return testData

def saveResult(result,csvName):
    with open(csvName,'w',newline='') as myFile:
        myWriter=csv.writer(myFile)
        for i in result:
            tmp=[]
            tmp.append(i)
            myWriter.writerow(tmp)

#调用scikit的knn算法包
from sklearn.neighbors import KNeighborsClassifier
def knnClassify(trainData,trainLabel,testData):
    knnClf=KNeighborsClassifier(n_neighbors=5)#default:k = 5,defined by yourself:KNeighborsClassifier(n_neighbors=10)
    knnClf.fit(trainData,ravel(trainLabel))
    testLabel=knnClf.predict(testData)
    saveResult(testLabel,'sklearn_knn_Result.csv')
    return testLabel

if __name__ == '__main__':
    trainData, trainLabel = load_train_data()
    testData = loadTestData()
    result = knnClassify(trainData, trainLabel, testData)