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

    def run(self):
        self.updateLabels()

        center = [0 for i in range(self.m)]
        s = [center[:] for i in range(self.k)]

        for item,label in zip(self.data,self.labels):
            for i in range(self.m):
                s[label][i]+= (item[i]/self.count[label])

        self.centroids = s

    def calcMaximumDistance(self):
        result = 0
        for i in range(0,self.n):
            for j in range(i,self.n):
                if self.labels[i] == self.labels[j]:
                    result = max(result,self.distance(self.data[i],self.data[j]))
        return result

    def solve(self):
        self.centroids = self.pickRandomCentroids()
        for i in range(0,10):
            self.run()
        return {"labels" : self.labels,
                "maximum distance" : self.calcMaximumDistance()}
