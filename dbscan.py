import numpy as np
from kmeans import pairwise_dist


class DBSCAN(object):
    def __init__(self, eps, minPts, dataset):
        self.eps = eps
        self.minPts = minPts
        self.dataset = dataset

    def fit(self):
        """Fits DBSCAN to dataset and hyperparameters defined in init().
        Args:
            None
        Return:
            cluster_idx: (N, ) int numpy array of assignment of clusters for each point in dataset
        Hint: Using sets for visitedIndices may be helpful here.
        Iterate through the dataset sequentially and keep track of your points' cluster assignments.
        If a point is unvisited or is a noise point (has fewer than the minimum number of neighbor points), then its cluster assignment should be -1.
        Set the first cluster as C = 0
        """
        #this isn't right. Partial points?

        visitedIndices = set()

        cluster_idx = np.zeros(self.dataset.shape[0], dtype=int)

        D = 0
    
        for index in range(self.dataset.shape[0]):

            if index not in visitedIndices:
                
                visitedIndices.add(index)
                neighborIndices = self.regionQuery(index)
            
                if len(neighborIndices) < self.minPts:
                    cluster_idx[index] = -1

                else:
                    D += 1
                    cluster_idx[index] = D
                    self.expandCluster(index, neighborIndices, D, cluster_idx, visitedIndices)
    
        return cluster_idx
        # raise NotImplementedError

    def expandCluster(self, index, neighborIndices, C, cluster_idx, visitedIndices):
        """Expands cluster C using the point P, its neighbors, and any points density-reachable to P and updates indices visited, cluster assignments accordingly
           HINT: regionQuery could be used in your implementation
        Args:
            index: index of point P in dataset (self.dataset)
            neighborIndices: (N, ) int numpy array, indices of all points witin P's eps-neighborhood
            C: current cluster as an int
            cluster_idx: (N, ) int numpy array of current assignment of clusters for each point in dataset
            visitedIndices: set of indices in dataset visited so far
        Return:
            None
        Hints: 
            np.concatenate(), np.unique(), np.sort(), and np.take() may be helpful here
            A while loop may be better than a for loop
        """
        #also not right but this is what I have

        i = 0
        
        while i < len(neighborIndices):
            neighborIndex = neighborIndices[i]

            if neighborIndex not in visitedIndices:

                visitedIndices.add(neighborIndex)
                neighborNeighborIndices = self.regionQuery(neighborIndex)

                if len(neighborNeighborIndices) >= self.minPts:
                    neighborIndices = np.concatenate((neighborIndices, neighborNeighborIndices))
        
            if cluster_idx[neighborIndex] == 0:
                cluster_idx[neighborIndex] = C
        
            i += 1
        # raise NotImplementedError

    def regionQuery(self, pointIndex):
        """Returns all points within P's eps-neighborhood (including P)

        Args:
            pointIndex: index of point P in dataset (self.dataset)
        Return:
            indices: (I, ) int numpy array containing the indices of all points within P's eps-neighborhood
        Hint: pairwise_dist (implemented above) and np.argwhere may be helpful here
        """
        #I hate this
        #what do I do with pointIndex?

        regionDist = pairwise_dist(self.dataset[pointIndex, np.newaxis], self.dataset)

        indices = np.argwhere(regionDist[0] <= self.eps).flatten()

        return indices
    
        # raise NotImplementedError
