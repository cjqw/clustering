#!env /usr/bin/python3

import numpy as np
from random import randint
from math import *
from kmeans import *

def parseItem(item):
    dims = item.split(',')
    return [float(dim) for dim in dims]

def readData():
    inputFile = 'data/data.csv'
    with open(inputFile,'r') as fin:
        data = fin.readlines()[1:]
    return [parseItem(item) for item in data]

def distance(x,y):
    return sqrt((x[0]-y[0])**2
                +(x[1]-y[1])**2
                +(x[2]-y[2])**2)

if __name__ == "__main__":
    data = readData()
    ans = 100
    for i in range(100):
        label = KMeans(data,3,distance).solve()
        maximumDistance = label["maximum distance"]
        print(maximumDistance)
        ans = min(ans,maximumDistance)
    print(ans)
