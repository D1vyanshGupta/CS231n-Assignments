import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient

  # get the number of classes
  num_classes = W.shape[1]

  # get the number of training examples
  num_train = X.shape[0]

  # initialize loss to 0.0
  loss = 0.0

  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        dW[:, j] -= (np.sum(1 * ((scores - correct_class_score + 1) > 0)) - 1) * X[i, :]
        continue

      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] += X[i, :]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


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

  # get number of training examples.
  num_train = X.shape[0]

  # calculate the score matrix
  # ith row gives the score vector for the ith data point
  score_matrix = X.dot(W)

  # for each score vector, get the score corresponding to the correct class
  # i.e. from each row of the matrix, use the y value as an index to get the correct class score
  correct_class_scores = score_matrix[np.arange(num_train), y]

  # for all entries in each row of the matrix, subtract the score of the correct class and add 1
  # this is for the hinge loss calculation
  margin_matrix = score_matrix - correct_class_scores[:, np.newaxis] + 1

  # for all score vectors, the contribution of the correct class towards the hinge loss is 0
  margin_matrix[np.arange(num_train), y] = 0

  # average the hinge losses across all data points
  loss = np.sum(margin_matrix[margin_matrix > 0]) / num_train

  # implement L2 regularization on W
  loss += reg * np.sum(W * W)
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

  # to calculate derivative of hinge loss function w.r.t all j != y_i
  # set all entries greater than 0, to the value 1
  margin_matrix = 1 * (margin_matrix > 0)

  # to calculate derivative of hinge loss function w.r.t y_i
  # if the hinge loss > 0, then set to -1, else it remains 0
  # then sum all non-zero terms to calculate the derivative
  margin_matrix[np.arange(num_train), y] = (-1) * np.sum(margin_matrix, axis=1)

  # this is to implement the 2nd back propagation step
  dW = (X.T).dot(margin_matrix)

  # average the gradient values across all data points
  dW /= num_train

  # derivative of the L2 regularization term
  dW += 2 * reg * W

  return loss, dW
