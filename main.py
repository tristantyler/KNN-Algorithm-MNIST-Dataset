import pandas as pd
import time
import math
import matplotlib.pyplot as plt
import pickle
import os


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
    ls = convertToNodes(ls)
    graphSetup()
    xlist = []
    ylist = []

    for node in ls:
        xlist.append(node.x)
        ylist.append(node.y)

    plt.plot(xlist, ylist, 'ks')
    plt.show()


def convertToNodes(data):
    """Takes a dataframe object and converts them to a list of nodes with values and their x,y coords
    on a 28x28 image"""
    nodelist = []

    for i, node in enumerate(data):
        if int(node) > 1:
            x = (i % 28)
            y = int(i / 28)
            nodelist.append(Node(x, y, int(node)))
    return nodelist


def convertToList2(data):
    return [list(data.iloc[row]) for row in range(len(data))]


def convertToList5(data, traintrue=False):
    betterlist = []
    nl = [list(data.iloc[row]) for row in range(len(data))]
    for row in nl:
        if traintrue:
            betterrow = [row[0]]
        else:
            betterrow = []
        for count, node in enumerate(row):
            if node != 0:
                betterrow.append(count)
        betterlist.append(betterrow)

    return betterlist


def calculateDistance2(testrow, traindata, k):
    distanceList = []  # holds the distance calculation for each train data row
    indlist = []  # holds indexes for the k number of closest train data rows
    dlist = []  # holds the distance value for the k number of closest train data rows
    endlist = [0 for x in range(10)]  # holds total weighted votes for each class based on index in list

    for td in traindata:  # calculates distance for each train data row from testrow
        a = len(set(testrow).intersection(td))
        b = abs(len(testrow) - a) + abs(len(td) - a)
        distanceList.append(1 / b)

    for i in range(k):  # gets the k number of closest train data rows
        ind = distanceList.index(max(distanceList))
        indlist.append(ind)
        dlist.append(max(distanceList))
        distanceList[ind] = 0
    tot = sum(dlist)

    for i in range(k):  # assigns votes to respective classes
        endlist[traindata[indlist[i]][0]] += dlist[i] / tot

    return endlist.index(max(endlist))  # returns the value of the max in list, i.e the class


def prepareData(skip=False):
    if skip:
        print("Converting Train Data to List...")
        trainnodelist = convertToList5(pd.read_csv("MNIST_train.csv"), True)
        print("Converting Test Data to List...")
        testnodelist = convertToList5(pd.read_csv("MNIST_test.csv"), True)

        return trainnodelist, testnodelist

    if not os.path.exists("knn.pckl"):
        print("Making Pickle...")
        print("Converting Correct Data to List...")
        corlist = convertToList2(pd.read_csv("submission.csv"))
        print("Converting Train Data to List...")
        trainnodelist = convertToList5(pd.read_csv("train.csv"), True)
        print("Converting Test Data to List...")
        testnodelist = convertToList5(pd.read_csv("test.csv"))

        with open("knn.pckl", "wb") as file:
            pickle.dump((corlist, trainnodelist, testnodelist), file)
    else:
        print("Getting Pickle..")
        with open("knn.pckl", "rb")as file:
            corlist, trainnodelist, testnodelist = pickle.load(file)

    return corlist, trainnodelist, testnodelist


def main():
    start = time.time()
    #corlist, trainnodelist, testnodelist = prepareData()
    trainnodelist, testnodelist = prepareData(True)

    count = 0
    #print("Took ", time.time() - start, "seconds")
    start = time.time()
    print("Computing classes...")
    for j, i in enumerate(testnodelist):
        startcomp = time.time()

        calcvalue = calculateDistance2(i, trainnodelist, 9)
        #expected = corlist[j][1]
        expected = i[0]
        print("Desired class:", expected, "Computed class:", calcvalue, "| Took", round((time.time() - startcomp), 4),
              "seconds")
        if expected == calcvalue:
            count += 1
        print("{0}%".format(round((count / (j + 1) * 100), 4)), j + 1)
    print("{0}%".format(round((count / len(testnodelist)) * 100), 4))

    print("Took ", time.time() - start, "seconds")


main()
