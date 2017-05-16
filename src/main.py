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

# Setting
number_of_iteration = 100
print_raw_data = False
number_of_clusters = 4
print_the_result = True

# Main Procedure
if __name__ == "__main__":
    data,label = readData()
    if print_raw_data:
        draw(data,label)

    ans = {'maximum distance':9999,"labels":[]}
    maxDistance = getItem('maximum distance')
    avg,maximum_max_distance = 0,0

    for i in range(number_of_iteration):
        result = KMeans(data,number_of_clusters,distance).solve()
        avg += maxDistance(result)
        if(maxDistance(ans) > maxDistance(result)):
            ans = result
        if(maxDistance(result) > maximum_max_distance):
            maximum_max_distance = maxDistance(result)

    avg /= number_of_iteration
    print("Minimum max distance:",maxDistance(ans))
    print("Maximum max distance:",maximum_max_distance)
    print("Average max distance:",avg)
    if print_the_result:
        draw(data,ans['labels'])
