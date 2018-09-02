from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################

    # implementing the forward pass

    # calculate the intermediate signal matrix for the first layer
    # ith row gives the intermediate signal vector for the ith data point
    signal_l1 = X.dot(W1) + b1

    # apply the ReLU activation function to the intermediate signal vectors for all data points
    signal_l1[signal_l1 < 0] = 0

    # calculate the intermediate signal matrix for the second layer
    # ith row gives the intermediate signal vector for the ith data point
    signal_l2 = signal_l1.dot(W2) + b2

    # calculate the final scores after the second layer
    scores = signal_l2

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################

    # exponentiate all the entries of every signal vector
    exp_signal = np.exp(signal_l2)

    # for every signal vector, calculate the posterior probability of correct classification
    pos_prob = exp_signal[np.arange(N), y] / np.sum(exp_signal, axis=1)

    # calculate the log-likelihood of correct classification
    log_lhood = np.log(pos_prob)

    # calculate the sum of losses for every data point
    loss = -np.sum(log_lhood)

    # average the loss across all data points
    loss /= N

    # implement L2 regularization on W1 and W2
    loss += reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    # first back propagation

    # divide every component of every exp_signal vector by the sum of the components
    exp_signal /= np.sum(exp_signal, axis=1)[:, np.newaxis]

    # subtract 1 from the posterior probability of correct classification
    exp_signal[np.arange(N), y] -= 1

    # second back propagation

    # calculate the gradient w.r.t. b2
    db2 = np.sum(exp_signal, axis=0)

    # average the gradient across all data points
    db2 /= N

    # third back propagation

    # calculate the gradient w.r.t. W2
    dW2 = (signal_l1.T).dot(exp_signal)

    # average the gradient across all data points
    dW2 /= N

    # account for L2 regularization while calculating the gradient
    dW2 += 2 * reg * W2

    # fourth and fifth back propagation

    signal_l1[signal_l1 > 0] = 1

    # sixth back propagation

    # calculate the gradient w.r.t. b1
    db1 = np.sum(exp_signal.dot(W2.T) * signal_l1, axis=0)

    # average the gradient across all data points
    db1 /= N

    # seventh back propagation

    # calculate the gradient w.r.t. W2
    dW1 = (X.T).dot(exp_signal.dot(W2.T) * signal_l1)

    # average the gradient across all data points
    dW1 /= N

    # account for L2 regularization while calculating the gradient
    dW1 += 2 * reg * W1

    # put values in the dictionary
    grads['b2'] = db2
    grads['W2'] = dW2
    grads['b1'] = db1
    grads['W1'] = dW1

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################

      # randomly sample indices from the training data
      indices = np.random.choice(num_train, size=batch_size, replace=True)
      X_batch = X[indices]
      y_batch = y[indices]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################

      # implement gradient descent
      self.params['W1'] -= learning_rate * grads['W1']
      self.params['W2'] -= learning_rate * grads['W2']
      self.params['b1'] -= learning_rate * grads['b1']
      self.params['b2'] -= learning_rate * grads['b2']
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################

    # implement forward pass for the test data

    # forward pass through the first layer
    signal_l1 = X.dot(self.params['W1']) + self.params['b1']

    # ReLU activation after the first layer
    signal_l1[signal_l1 < 0] = 0

    # forward pass through the second layer
    signal_l2 = signal_l1.dot(self.params['W2']) + self.params['b2']

    # retrieve class labels corresponding to largest posterior probability
    y_pred = np.argmax(signal_l2, axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred
