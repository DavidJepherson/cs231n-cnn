import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  """
  #########meine loesung
  num_classes = W.shape[0]
  num_train = X.shape[1]
  entropy = np.zeros([1,num_classes])
  for i in xrange(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
        entropy += np.exp(scores[j])
    loss[i] =  -(correct_class_score) + np.log(np.sum(entropy))
  loss = np.sum(loss)/num_train
 
  #### erste loesung
  ### loss
  scores = W.dot(X)
  scores_max = np.max(scores, axis=0)
  scores -= scores_max
  exp_scores = np.exp(scores)
  sums = np.sum(exp_scores, axis=0)
  log_sums = np.log(sums)
  scores_y = np.array([scores[y[i], i] for i in xrange(X.shape[1])])
  loss = np.sum(-scores_y + log_sums)

  loss /= X.shape[1]
  loss += .5 * np.sum(W * W)

  ### dW
  for i in xrange(X.shape[1]):
    # dW += 1./sums[i] * log_sums[i] * X[:, i]
    dW += 1./sums[i] * exp_scores[:, i].reshape(-1, 1) * X[:, i]
    dW[y[i]] -= X[:, i]

  dW /= X.shape[1]
  dW += reg * W
  """
  """ kireeti
      scores = W.dot(X)
    #scores -= np.max(scores)
        
    N = X.shape[1]
    for j in xrange(N):
        loss += (-1 *(scores[y[j],j])) + np.log(np.sum(np.exp(scores[:,j]))) + reg*(np.sum(np.sum(W)))    
    loss = float(loss)/N
    print 'Loss: ', loss
    dW = analytic_gradient(W, X, y, reg)

  """
  #### zweite loesung und am besten
  C = W.shape[0]
  N = X.shape[1]
  scores = np.dot(W,X)
  scores -= np.max(scores)
  correct_class_scores = scores[y,np.arange(N)]
  sumexp = np.sum(np.exp(scores),axis=0)
  loss_i = -correct_class_scores + np.log(sumexp)
  loss = np.sum(loss_i) / N
  loss += 0.5 * reg * np.sum(W*W)

  for i in xrange(N):
    dW_i = np.exp(scores[:,i]) / sumexp[i]
    dW_i[y[i]] -= 1
    dW += np.outer(dW_i, X[:,i])

  dW /= N
  dW += reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)



  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  #############erste loesung und am besten
  C = W.shape[0]
  N = X.shape[1]
  # ## BACKPROP METHOD
  scores = np.dot(X.T, W.T)
  expsc = np.exp(scores)
  sumsc = np.sum(expsc, axis=1)
  logsc = np.log(sumsc)
  softmax = -scores[np.arange(N), y] + logsc
  loss = np.sum(softmax)/N + 0.5*reg*(np.sum(np.sum(W)))

  # Initialize dscores to zero
  dscores = np.zeros_like(scores)

  # backprop loss
  dsoftmax = np.ones(N)/N
  dW += reg * W
  # backprop softmax
  dscores[np.arange(N), y] -= dsoftmax
  dlogsc = dsoftmax
  # backprop logsc
  dsumsc = dlogsc / sumsc
  # backprop sumsc
  dexpsc = np.outer(dsumsc, np.ones(C))
  # backprop expsc
  dscores += np.exp(scores) * dexpsc
  # backprop scores
  dW += np.dot(dscores.T, X.T)


  # scores = np.dot(W,X)
  # scores -= np.max(scores)
  # correct_class_scores = scores[y,np.arange(N)]
  # sumexp = np.sum(np.exp(scores),axis=0)
  # loss_i = -correct_class_scores + np.log(sumexp)
  # loss = np.sum(loss_i) / N
  # loss += 0.5 * reg * np.sum(W*W)

  # dW = np.exp(scores) / sumexp
  # dW[y, np.arange(N)] -= 1
  # dW = np.dot(dW,X.T)

  # dW /= N
  # dW += reg * W
  ###### endn of erste loesung

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
