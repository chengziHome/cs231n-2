from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        self.params['W1'] = weight_scale * np.random.randn(input_dim,hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim,num_classes)
        self.params['b2'] = np.zeros(num_classes)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']

        hidden_data,hidden_cache= affine_relu_forward(X.reshape(X.shape[0],-1), W1, b1)
        scores,_ = affine_forward(hidden_data,W2,b2)


        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}


        loss,dscores = softmax_loss(scores,y)
        loss += 0.5 * self.reg * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

        dX2,dW2,db2 = affine_backward(dscores,(hidden_data,W2,b2))
        _,dW1,db1 = affine_relu_backward(dX2,hidden_cache)
        grads['W1'] = dW1 + self.reg * W1
        grads['b1'] = db1
        grads['W2'] = dW2 + self.reg * W2
        grads['b2'] = db2


        # loss, dscores = softmax_loss(scores, y)
        # loss = loss + 0.5 * self.reg * np.sum(self.params['W1'] * self.params['W1']) + 0.5 * self.reg * np.sum(
        #     self.params['W2'] * self.params['W2'])
        # dx2, dw2, db2 = affine_backward(dscores, a2_cache)
        # grads['W2'] = dw2 + self.reg * self.params['W2']
        # grads['b2'] = db2
        # # dx2_relu = relu_backward(dx2, r1_cache)
        # # dx1, dw1, db1 = affine_backward(dx2_relu, a1_cache)
        # dx1, dw1, db1 = affine_relu_backward(dx2, ar1_cache)
        # grads['W1'] = dw1 + self.reg * self.params['W1']
        # grads['b1'] = db1

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}


        for i,hidden_dim in enumerate(hidden_dims):
            self.params['W'+str(i+1)] = weight_scale * np.random.randn(input_dim,hidden_dim)
            self.params['b'+str(i+1)] = np.zeros(hidden_dim)
            if self.use_batchnorm:
                self.params['gamma%d' % (i + 1)] = np.ones(hidden_dim)
                self.params['beta%d' % (i + 1)] = np.zeros(hidden_dim)
            input_dim = hidden_dim
        self.params['W'+str(self.num_layers)] = weight_scale * np.random.randn(input_dim,num_classes)
        self.params['b'+str(self.num_layers)] = np.zeros(num_classes)

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None



        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################

        # input = X
        # out = None
        # cache_list = []
        # for i in range(self.num_layers - 1):
        #     W_i = self.params['W' + str(i + 1)]
        #     b_i = self.params['b' + str(i + 1)]
        #     out, cache = affine_relu_forward(input, W_i, b_i)
        #     cache_list.append(cache)
        #     input = out
        #
        # W_ = self.params['W' + str(self.num_layers)]
        # b_ = self.params['b' + str(self.num_layers)]
        #
        # scores, cache_ = affine_forward(input, W_, b_)

        layer_input = X
        ar_cache = {}
        dp_cache = {}

        for lay in range(self.num_layers - 1):
            if self.use_batchnorm:
                layer_input, ar_cache[lay] = affine_bn_relu_forward(layer_input,
                                                                    self.params['W%d' % (lay + 1)],
                                                                    self.params['b%d' % (lay + 1)],
                                                                    self.params['gamma%d' % (lay + 1)],
                                                                    self.params['beta%d' % (lay + 1)],
                                                                    self.bn_params[lay])
            else:
                layer_input, ar_cache[lay] = affine_relu_forward(layer_input, self.params['W%d' % (lay + 1)],
                                                                 self.params['b%d' % (lay + 1)])

            if self.use_dropout:
                layer_input, dp_cache[lay] = dropout_forward(layer_input, self.dropout_param)

        ar_out, ar_cache[self.num_layers] = affine_forward(layer_input, self.params['W%d' % (self.num_layers)],
                                                           self.params['b%d' % (self.num_layers)])
        scores = ar_out

        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}

        # loss,dscores = softmax_loss(scores,y)
        # reg_loss = 0.0
        # for i in range(self.num_layers):
        #     W_i = self.params['W'+str(i+1)]
        #     reg_loss += np.max(np.square(W_i))
        # loss += 0.5 * self.reg * reg_loss
        #
        # x,w,b = cache_
        # dx,dw,db = affine_backward(dscores,cache_)
        # grads['W'+str(self.num_layers)] = dw + self.reg * w
        # grads['b'+str(self.num_layers)] = db
        # dout = dx
        # for i in range(self.num_layers-1):
        #     cache = cache_list[self.num_layers-2-i]
        #     dx,dw,db = affine_relu_backward(dout,cache)
        #     grads['W' + str(self.num_layers-1-i)] = dw + self.reg * self.params['W'+str(self.num_layers-1-i)]
        #     grads['b' + str(self.num_layers-1-i)] = db
        #     dout = dx


        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        loss, dscores = softmax_loss(scores, y)
        dhout = dscores
        loss = loss + 0.5 * self.reg * np.sum(
            self.params['W%d' % (self.num_layers)] * self.params['W%d' % (self.num_layers)])
        dx, dw, db = affine_backward(dhout, ar_cache[self.num_layers])
        grads['W%d' % (self.num_layers)] = dw + self.reg * self.params['W%d' % (self.num_layers)]
        grads['b%d' % (self.num_layers)] = db
        dhout = dx
        for idx in range(self.num_layers - 1):
            lay = self.num_layers - 1 - idx - 1
            loss = loss + 0.5 * self.reg * np.sum(self.params['W%d' % (lay + 1)] * self.params['W%d' % (lay + 1)])
            if self.use_dropout:
                dhout = dropout_backward(dhout, dp_cache[lay])
            if self.use_batchnorm:
                dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(dhout, ar_cache[lay])
            else:
                dx, dw, db = affine_relu_backward(dhout, ar_cache[lay])
            grads['W%d' % (lay + 1)] = dw + self.reg * self.params['W%d' % (lay + 1)]
            grads['b%d' % (lay + 1)] = db
            if self.use_batchnorm:
                grads['gamma%d' % (lay + 1)] = dgamma
                grads['beta%d' % (lay + 1)] = dbeta
            dhout = dx

        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
