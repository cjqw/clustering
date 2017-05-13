#!env /usr/bin/python3

import numpy as np
from math import *
from kmeans import KMeans
from util.util import *

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def parseLine(line):
    dims = line.split(',')
    return mapv(float,dims[:-1]), int(float(dims[3]))

def readData():
    inputFile = 'data/data.csv'
    with open(inputFile,'r') as fin:
        lines = fin.readlines()[1:]
    result = mapv(parseLine,lines)
    return mapv(getItem(0),result), mapv(getItem(1),result)

def distance(x,y):
    return sqrt((x[0]-y[0])**2
                +(x[1]-y[1])**2
                +(x[2]-y[2])**2)

def draw(data,label):
    model = partition(list(zip(data,label)),getItem(1))
    ax=plt.subplot(111,projection='3d')

    for label in model:
        position = mapv(getItem(0),model[label])
        xs = mapv(getItem(0),position)
        ys = mapv(getItem(1),position)
        zs = mapv(getItem(2),position)
        ax.scatter(xs,ys,zs)

    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()

if __name__ == "__main__":
    data,label = readData()
    # draw(data,label)
    ans = {'maximum distance':9999,"labels":[]}
    maxDistance = getItem('maximum distance')
    for i in range(100):
        result = KMeans(data,3,distance).solve()
        if(maxDistance(ans) > maxDistance(result)):
            ans = result
    print(ans)
    draw(data,ans['labels'])
