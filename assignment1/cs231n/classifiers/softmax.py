import numpy as np
from random import shuffle

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # get number of training examples
  num_train = X.shape[0]

  # get number of classes
  num_classes = W.shape[1]

  for i in range(num_train):
      scores = (X[i]).dot(W)
      scores -= np.max(scores)
      exp_scores = np.exp(scores)
      sum_exp_scores = np.sum(exp_scores)
      prob = exp_scores[y[i]] / sum_exp_scores
      loss -= np.log(prob)

      for j in range(num_classes):
          if j == y[i]:
              dW[:, j] -= ((sum_exp_scores / exp_scores[y[i]]) * (sum_exp_scores * exp_scores[y[i]] - np.square(exp_scores[y[i]])) / (np.square(sum_exp_scores))) * X[i, :]
          else:
              dW[:, j] += ((sum_exp_scores / exp_scores[y[i]]) * (exp_scores[y[i]] * exp_scores[j]) / (np.square(sum_exp_scores))) * X[i, :]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  loss += reg * np.sum(W * W)

  # Add regularization to the loss.
  dW /= num_train
  dW += 2 * reg * W

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

  # get the number of training examples
  num_train = X.shape[0]

  # calculate the score matrix
  # ith row gives the score vector for the ith data point
  scores = X.dot(W)

  # normalize all the entries of every score vector by subtracting from the largest entry
  # this is to ensure numerical stability while implementing the algorithm
  scores = scores - np.max(scores, axis=1)[:, np.newaxis]

  # exponentiate all the entries of every score vector
  exp_scores = np.exp(scores)

  # for every score vector, calculate the posterior probability of correct classification
  pos_prob = exp_scores[np.arange(num_train), y] / np.sum(exp_scores, axis=1)

  # calculate the log-likelihood of correct classification
  log_lhood = np.log(pos_prob)

  # calculate the sum of losses for every data point
  loss = -np.sum(log_lhood)

  # for each score vector, divide all entries by sum of the entries to calculate the posterior probabilities
  exp_scores /= np.sum(exp_scores, axis=1)[:, np.newaxis]

  # for each score vector, subtract 1 from the posterior probability corresponding to the correct class
  exp_scores[np.arange(num_train), y] -= 1

  # this is to implement the 2nd back propagation step
  dW = (X.T).dot(exp_scores)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  # average the loss across all data points
  loss /= num_train

  # implement L2 regularization on W
  loss += reg * np.sum(W * W)

  # average the gradient values across all data points
  dW /= num_train

  # derivative of the L2 regularization term
  dW += 2 * reg * W

  return loss, dW
