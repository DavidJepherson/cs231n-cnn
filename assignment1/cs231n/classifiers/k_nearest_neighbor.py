import numpy as np

class KNearestNeighbor:
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Input:
    X - A num_train x dimension array where each row is a training point.
    y - A vector of length num_train, where y[i] is the label for X[i, :]
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Input:
    X - A num_test x dimension array where each row is a test point.
    k - The number of nearest neighbors that vote for predicted label
    num_loops - Determines which method to use to compute distances
                between training points and test points.

    Output:
    y - A vector of length num_test, where y[i] is the predicted label for the
        test point X[i, :].
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Input:
    X - An num_test x dimension array where each row is a test point.

    Output:
    dists - A num_test x num_train array where dists[i, j] is the distance
            between the ith test point and the jth training point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      for j in xrange(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]               #
        #####################################################################
	dists[i,j] = np.sqrt(np.sum(np.square(X[i]-self.X_train[j]))) #usere loesung
	# dists[i,j] = np.sqrt(np.sum((X[i] - self.X_train[j]) ** 2)) #erste loesung
	# dists[i,j] = np.sqrt(((X[i,:]-self.X_train[j,:])**2).sum(axis=0))
        # dists[i,j] = np.sqrt(np.sum((X[i,:]-self.X_train[j,:])**2,axis=0))
	# dists[i,j] = np.sqrt(np.sum(np.square(X[i,:]-self.X_train[j,:]),axis=1))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      #dists[i,:] = np.linalg.norm(X[i,:]-self.X_train,axis = 1)
      dists[i,:]= np.sqrt(np.sum(np.square(X[i,:]-self.X_train),axis=1)) #unsere loesung
      #######
      # X[i].shape = (3072,) -> broadcast to fit X_train shape. erste loesung
      #dists[i] = np.sqrt(np.sum((X[i] - self.X_train) ** 2, axis=1))
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    #X1 = np.zeros((X.shape[0],1,X.shape[1]))
    #X1[:,0,:] = X
    #X_train1 = np.zeros((1,self.X_train.shape[0],self.X_train.shape[1]))
    #X_train1[0,:,:] = self.X_train
    #dists = np.linalg.norm(X_train1-X1,axis = 2)
    dists = np.sqrt((X**2).sum(axis=1)[:, np.newaxis] + (self.X_train**2).sum(axis=1) - 2 * X.dot(self.X_train.T))
    #####
    ## from scipy.spatial.distance import cdist
    ## dists = cdist(X, self.X_train, metric='euclidean')
    ## to fully vectorize, use of the formula: (a-b)^2 = a^2 + b^2 -2ab
    ## (a-b)^2 = quadra -2 * prod
    ## with quadra = a^2 + b^2; and prod = ab
    #a2 = np.sum(X ** 2, axis=1) # shape: (500,)
    #b2 = np.sum(self.X_train ** 2, axis=1) # shape: (5000,)
    #print a2.shape
    #aa2 = a2.reshape(a2.shape[0], 1) # reshape a2 to (500,1) to be able to broadcast a2 and sum to b2
    #print aa2.shape
    #quadra = aa2 + b2 # shape = (500, 5000)
    #prod = np.dot(X, self.X_train.T) # shape = (500, 5000)
    #dists = np.sqrt(aa2 + b2 -2 * np.dot(X, self.X_train.T)) # shape = (500, 5000)
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Input:
    dists - A num_test x num_train array where dists[i, j] gives the distance
            between the ith test point and the jth training point.

    Output:
    y - A vector of length num_test where y[i] is the predicted label for the
        ith test point.
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # training point, and use self.y_train to find the labels of these      #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      closest_y = np.array(self.y_train[np.argsort(dists[i,:],axis=0)[:k]]) #unsere loesung
      #closest_y = np.array(self.y_train[np.argsort(dists[i,:],axis=0)[:k]],dtype=float) #unsere loesung
      ####
      #ind = np.argsort(dists[i, :], axis=0) #erste loesung
      #closest_y = self.y_train[ind[:k]]
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      #d = {x:closest_y.count(x) for x in closest_y}
      #c, b = d.keys(), d.values()
      #e=c[b.index(max(b))]
      #y_pred[i] = self.y_train[closest_y]
      y_pred[i] = np.bincount(closest_y).argmax() # meine loesung worked

      ########### worked
      #from scipy.stats import mode # zweite loesung meine loesong
      #y_pred[i] = mode(closest_y)[0][0]
      ###############   zweite loesung worked but the accuracy is a disaster  
      #from collections import Counter
      ## pick nearest neighbors  
      #closest_y = self.y_train[np.argsort(dists[i]) < k]
      
      ## count which class appears most
      #y_pred[i] = Counter(closest_y).most_common(1)[0][0] 
      ############## erste loesung worked
      #types = np.unique(closest_y)
      ## print 'types', types
      #closest_y_list = closest_y.tolist()
      #occ = [closest_y_list.count(lab) for lab in types]
      ## print 'occ:', occ
      #y_pred[i] = types[np.argmax(occ)]
      ## print 'pred: ', y_pred[i]

      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred

# with the CIFAR dataset
# Two loop version took 80.492011 seconds
# One loop version took 112.854208 seconds
# No loop version took 13.636812 seconds

# accuracy with k=10:
# Got 141 / 500 correct => accuracy: 0.282000

# Final accuracy, after 5-fold cross-validation to choose best_k=50:
# Got 293 / 500 correct => accuracy: 0.586000

