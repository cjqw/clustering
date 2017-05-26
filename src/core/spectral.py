from random import randint
from core.kmeans import KMeans
import numpy as np

def normalDisFunction(a,b):
        return np.linalg.norm(a-b)

class SpectralClustering:
    def __init__(self,data,k,distanceFunction):
        self.data = np.array(data)
        self.k = k
        self.n = len(data)
        self.m = len(data[0])
        self.distance = distanceFunction

    def calculateLaplaceMatrix(self):
        L = np.zeros((self.n,self.n))
        for i in range(0,self.n):
            for j in range(i+1,self.n):
                L[i][j] = - self.distance(self.data[i],self.data[j])
                L[j][i] = L[i][j]
                L[i][i] -= L[i][j]
                L[j][j] -= L[i][j]
        self.LaplaceMatrix = L

    def calculateEigenVectorMatrix(self):
        eigval,eigvec = np.linalg.eig(self.LaplaceMatrix)
        eigvec = np.transpose(eigvec)
        sortedEig = sorted(list(zip(eigval,eigvec)))

        # for item in sortedEig:
        #     val,vec = item
        #     x = np.dot(val,vec)
        #     y = np.dot(self.LaplaceMatrix,vec)
        #     print(np.linalg.norm(x-y))

        self.kEigVec = [sortedEig[i][1] for i in range(0,self.k)]
        self.kEigVec = np.transpose(self.kEigVec)

    def solve(self):
        self.calculateLaplaceMatrix()
        self.calculateEigenVectorMatrix()
        result = KMeans(self.kEigVec,self.k,
                        normalDisFunction).solve()
        return result
