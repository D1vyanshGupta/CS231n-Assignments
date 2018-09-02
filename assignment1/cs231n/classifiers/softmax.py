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

  for i in np.arange(num_train):
      # calculate the score vector
      score_vector = (X[i, :]).dot(W)

      # normalize the entries of the score vector
      # this is no ensure mathematical stability during intensive calculations
      score_vector -= np.max(score_vector)

      # exponentiate all the entries of the score vector
      exp_score_vector = np.exp(score_vector)

      # divide all the entries of the score vector by sum of all entries to calculate the posterior probability of classification
      exp_score_vector /= np.sum(exp_score_vector)

      # get posterior probability of correct classification
      prob = exp_score_vector[y[i]]

      # calculate log-likelihood
      log_lhood = np.log(prob)

      # add negative log-likelihood to the loss
      loss += (-1) * log_lhood

      # derivative of loss (L_i) w.r.t. score vector (s_i) is dL_i_by_s_i
      # s_i[j] are components of the score vector (s_i)
      # if j != y_i
      #     dL_i_by_s_i[j] = probability that X[i] belongs to class j = exp_score_vector[j]
      #   else
      #       dL_i_by_s_i[j] = probability that X[i] belongs to class y_i - 1 = exp_score_vector[y_i] - 1

      # dL_i_by_W_[k,l] = X[i, k] * dL_i_by_s_i[l]
      # => dL_i_by_W_[:,l] = X[i, :] * dL_i_by_s_i[l]

      # iterate through the entries of the score vector
      for j in np.arange(num_classes):
          if j == y[i]:
              # calculate dL_i_by_s_i[y_i]
              derivative = exp_score_vector[y[i]] - 1

              # calculate the contribution of s_i[y_i] to dL_i_by_W_[:,j]
              dW[:, j] += derivative * X[i, :]

          else:
              # calculate dL_i_by_s_i[y_i]
              derivative = exp_score_vector[j]

              # calculate the contribution of s_i[y_i] to dL_i_by_W_[:,j]
              dW[:, j] += derivative * X[i, :]

  # average the loss across all data points
  loss /= num_train

  # implement L2 regularization on W
  loss += reg * np.sum(W * W)

  # average the gradient across all data points
  dW /= num_train

  # account for the derivative of the L2 regularization term
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

  # average the loss across all data points
  loss /= num_train

  # implement L2 regularization on W
  loss += reg * np.sum(W * W)

  # average the gradient across all data points
  dW /= num_train

  # account for the derivative of the L2 regularization term
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
