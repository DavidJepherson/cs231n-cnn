import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  """
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  delta = 1.0

  for i in xrange(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + delta # note delta = 1
      if margin > 0:
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  """
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  #
  """
  #############meine loesung
  for i in xrange(num_train):
	ddW = np.zeros(W.shape)
	scores = W.dot(X[:, i])
	correct_class_score = scores[y[i]]
	for j in xrange(num_classes):
		if j == y[i]:
			continue
		margin = scores[j] - correct_class_score + delta
		if margin > 0:
      		  loss += margin
		#loss += max(0, margin)
		  ddW[j] += X[:,i] # the error btw analytical and numerical in this approach is too much because   ddWyi += ddW[j] ddW[y[i]] = -ddWyi weren't done.
   	dW += ddW
  loss /= num_train
  dW /= num_train
  

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  #
  """
  """
  ############## erste loesung

  for i in xrange(num_train):
  	ddW = np.zeros(W.shape)
  	ddWyi = np.zeros(W[0].shape)     ###?????????????????
  	scores = W.dot(X[:, i])
  	correct_class_score = scores[y[i]]
  	for j in xrange(num_classes):
  		if j == y[i]:
        		continue
      		margin = scores[j] - correct_class_score + delta
     		if margin > 0:
      		  loss += margin
      		  ddW[j] = X[:, i] ## be careful, it's a reference
      		  ddWyi += ddW[j] #important to get correct results
  	ddW[y[i]] = -ddWyi #important to get correct results
   	dW += ddW
  
  # divided by num_train
  loss /= num_train
  dW /= num_train
  
  # add regularization term
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  """
  """
  def svm_loss_naive(W, X, y, reg):

  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
 
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = np.zeros(1)
  for i in xrange(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    init = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
	init +=1
        dW[j,:] += X[:,i]
    dW[y[i],:] += -init*X[:,i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train  
  dW /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  """
  ### zweite und am besten loesong

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]

  grad_helper = np.zeros([num_classes, num_train])
  loss = 0.0
  for i in xrange(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        grad_helper[j,i] += 1
        grad_helper[y[i],i] -= 1

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  
  dW = grad_helper.dot(X.T)
  dW /= num_train
  dW += reg * W
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  """
  delta = 1
  scores = W.dot(X)
  margins = np.maximum(0, scores - scores[y] + delta)
  margins[y] = 0
  loss = np.sum(margin)
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  """
  """
  ##########erste loesung
  ### loss
  # wxy chooses scores of right labels
  # its shape is (#samples,)
  wx = W.dot(X)
  delta = 1
  wxy = [ wx[y[i], i] for i in xrange(wx.shape[1]) ]
  
  # judge expression
  # remember to exclude on y[i]'s
  judge = wx - wxy + delta
  # make judge 0 on y[i]
  for i in xrange(wx.shape[1]):
    judge[y[i], i] = 0
  
  # mass is a matrix holding all useful temp results
  # shape of judge is (#class, #train)
  mass = np.maximum(0, judge)
  
  loss = np.sum(mass) / X.shape[1]
  loss += 0.5 * reg * np.sum(W * W)
  """
  """
  scores = W.dot(X)
  loss = np.zeros(scores.shape)
  init = np.zeros(scores.shape)
  loss = scores - np.choose(y,scores)
  loss[np.where(loss!=0)] += 1 
  loss[np.where(loss<=0)] = 0
  init[np.where(loss>0)] = 1
  loss1 = loss
  loss = np.sum(loss)
  loss /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)  
  """
  ### zweite und am besten loesung erste part
  WX = W.dot(X)
  num_classes = W.shape[0]
  num_train = X.shape[1]
  margin = WX - WX[y, np.arange(num_train)] + 1
  margin[y, np.arange(num_train)] = 0
  margin[margin < 0] = 0
  loss = sum(sum(margin))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  """
  scores = W.dot(X)
  margins = np.maximum(0, scores - scores[y] + delta)
  ddW = np.sum(X,axis=1)
  margins[y] = 0
  ddW[y] -= ddW
  dW = np.sum(ddW)
  loss = np.sum(margin)
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  
  # add regularization term
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  """
  """
  ### erste loesung
    # continue on last snippet
  # weight to be producted by X
  # its shape is (#classes, #samples)
  weight = np.array((judge > 0).astype(int))
  
  # weights on y[i] needs special care
  weight_yi = -np.sum(weight, axis=0)
  for i in xrange(wx.shape[1]):
    weight[y[i], i] = weight_yi[i]
  
  # half vectorized
  for i in xrange(X.shape[1]):
    ddW = X[:, i] * weight[:, i].reshape(-1, 1)
    dW += ddW
    
  dW /= X.shape[1]
  dW += reg * W
  """
  """
  k1 = np.sum(init,axis=0)
  k2 = -k1*X
  dW = np.tile(np.sum(X,axis=1),(num_classes,1))
  #np.add.at(dW,y,-kT-XT)
  #np.add.at(dW,y,k2.transpose())
  #y2 = np.repeat(np.arange(num_classes),num_train).reshape(num_classes,num_train)
  #np.add.at(dW,np.int_(y2),X.transpose())
  #np.add.at(dW,y,-X.transpose())
  #dW /= num_train
  """
  ### zweite und am besten loesung zweite part
  grad = np.zeros([W.shape[0], X.shape[1]])
  grad_wj = margin > 0
  grad_y = -1*sum(grad_wj)
  grad = grad + grad_wj
  grad[y, np.arange(num_train)] = grad_y
  dW = grad.dot(X.T)
  dW /= num_train
  dW += reg * W
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
