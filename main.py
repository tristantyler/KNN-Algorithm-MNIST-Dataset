import pandas as pd
import time
import math
import matplotlib.pyplot as plt


class Node:
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value

    def __str__(self):
        return "(%i, %i) with value of %i" % (self.x, self.y, self.value)


def graphSetup():
    axes = plt.gca()
    axes.set_xlim([0, 28])
    axes.set_ylim([0, 28])
    axes.invert_yaxis()


def graph(ls):
    """Takes the x and y values from nodes and makes pairs and graphs them. Takes a node list."""
    graphSetup()
    xlist = []
    ylist = []

    for node in ls[1:]:
        xlist.append(node.x)
        ylist.append(node.y)

    plt.plot(xlist, ylist, 'ks')
    plt.show()


def convertToList(data):
    """Takes a dataframe object and converts them to a list of nodes with values and their x,y coords
    on a 28x28 image"""
    nodelist = []

    for row in range(len(data)):
        dt = list(data.iloc[row])
        dtl = [dt[0]]
        for i, node in enumerate(dt[1:]):
            if int(node) > 1:
                x = (i % 28)
                y = int(i / 28)
                dtl.append(Node(x, y, int(node)))
        nodelist.append(dtl)
    return nodelist


def convertToList2(data):
    nodelist = []
    for row in range(len(data)):
        dt = list(data.iloc[row])
        nodelist.append(dt)
    return nodelist


def calculateDistance(testrow, traindata, k):
    distanceList = []  # holds the distance calculation for each train data row
    indlist = []  # holds indexes for the k number of closest train data rows
    dlist = []  # holds the distance value for the k number of closest train data rows
    tot = 0  # sums of the k number of closest distance values
    endlist = [0 for x in range(10)]  # holds total weighted votes for each class based on index in list

    for td in traindata:  # calculates distance for each train data row from testrow
        total = 0
        for i in range(1, len(testrow)):
            if testrow[i] == 0 and td[i] == 0:
                pass
            elif testrow[i] == 0:
                total += 1
            elif td[i] == 0:
                total += 1
        distanceList.append(1 / (math.sqrt(total)))

    for i in range(k):  # gets the k number of closest train data rows
        ind = distanceList.index(max(distanceList))
        indlist.append(ind)
        dlist.append(max(distanceList))
        tot += max(distanceList)
        distanceList.pop(ind)

    for i in range(k):  # assigns votes to respective classes
        endlist[traindata[indlist[i]][0]] += dlist[i] / tot

    return endlist.index(max(endlist))  # returns the value of the max in list, i.e the class


def main():
    start = time.time()
    traindata = pd.read_csv("MNIST_train.csv")
    testdata = pd.read_csv("MNIST_test.csv")

    trainnodelist = convertToList2(traindata)
    testnodelist = convertToList2(testdata)

    count = 0
    for i in testnodelist:
        calcvalue = calculateDistance(i, trainnodelist, 7)
        print("Desired class:", i[0], "Computed class:", calcvalue)
        if i[0] == calcvalue:
            count += 1
    print("{0}%".format((count / len(testnodelist)) * 100))

    print("Took ", time.time() - start, "seconds")


main()
