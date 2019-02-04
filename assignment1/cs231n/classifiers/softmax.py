import numpy as np
from random import shuffle
from past.builtins import xrange
import math

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  num_classes = W.shape[1]

  scores = X.dot(W)

  for i in range(num_train):
    sum = 0.0
    for j in range(num_classes):
      sum += math.exp(scores[i,j])
    loss_i = - math.log(math.exp(scores[i,y[i]])/sum)
    loss += loss_i
    for j in range(num_classes):
      if j == y[i]:
        dW[:,j] += (-1/num_train) * (sum - math.exp(scores[i,y[i]]))/sum * X[i,:].T
      else:
        dW[:,j] += (1/num_train) * (math.exp(scores[i,j])/sum) * X[i,:].T

  loss = loss/num_train + 0.5 * reg * np.sum(W*W)

  dW += reg * W

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
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

  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = X.dot(W)
  normalized_scores = np.exp(scores) / np.sum(np.exp(scores),axis=1).reshape(-1,1)

  loss = (-1/num_train) * np.sum(np.log(normalized_scores[range(num_train),y])) + 0.5 * reg * np.sum(W*W)


  normalized_scores[range(num_train),y] -= 1
  coeff_mat = X.T.dot(normalized_scores)
  dW = coeff_mat/num_train + reg*W

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

