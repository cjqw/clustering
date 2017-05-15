from random import randint

class KMeans:
    def __init__(self,data,k,distanceFunction):
        self.data = data
        self.k = k
        self.n = len(data)
        self.m = len(data[0])
        self.distance = distanceFunction
        self.labels = [0 for i in range(self.n)]

    def pickRandomCentroids(self):
        centroids = []
        n = len(self.data)
        for i in range(0,self.k):
            sample = randint(0,n-1)
            while sample in centroids:
                sample = randint(0,n-1)
            centroids.append(sample)
        return [self.data[idx] for idx in centroids]

    def getLabel(self,item):
        distance = self.distance
        d,l = distance(item,self.centroids[0]) + 1, -1
        for centroid,label in zip(self.centroids,range(self.k)):
            dis = distance(item,centroid)
            if dis < d:
                d,l = dis,label
        return l

    def updateLabels(self):
        self.count = [0 for i in range(self.k)]
        for item,idx in zip(self.data,range(self.n)):
            label = self.getLabel(item)
            self.labels[idx] = label
            self.count[label] += 1

    def converge(self):
        eps = 0.0001
        self.updateLabels()

        center = [0 for i in range(self.m)]
        s = [center[:] for i in range(self.k)]

        for item,label in zip(self.data,self.labels):
            for i in range(self.m):
                s[label][i]+= (item[i]/self.count[label])

        s = sorted(s)
        result = True
        for x,y in zip(s,self.centroids):
            if x[0] - y[0] > eps or x[1] - y[1] > eps:
                result = False

        self.centroids = s
        return result

    def calcMaximumDistance(self):
        result = 0
        for i in range(0,self.n):
            for j in range(i,self.n):
                if self.labels[i] == self.labels[j]:
                    result = max(result,self.distance(self.data[i],self.data[j]))
        return result

    def solve(self):
        self.centroids = self.pickRandomCentroids()
        while not self.converge():
            pass
        return {"labels" : self.labels,
                "maximum distance" : self.calcMaximumDistance()}
