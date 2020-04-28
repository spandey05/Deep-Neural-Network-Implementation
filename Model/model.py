import numpy as np


def sigmoid(z, keep_prob):
    """
    Apply sigmoid activation to all elements of a matrix element-wise.

    :param z: data to apply sigmoid to
    :param keep_prob: the probability of each node to be included in the propagation
    :return:
        a: the result obtained on apply sigmoid activation
        activation_cache: stored value of mask matrix for dropout, and the data passed itself
    """
    assert isinstance(z, np.ndarray) and isinstance(keep_prob, float)
    assert 1. >= keep_prob > 0.

    a = 1. / (1. + np.exp(-z))
    d = np.random.random_sample(size=z.shape)
    d = (d < keep_prob).astype(int)
    a = (a * d) / keep_prob

    activation_cache = {
        "D": d,
        "Z": z,
    }
    assert z.shape == a.shape == d.shape
    return a, activation_cache


def relu(z, keep_prob):
    """
    Apply relu activation to all elements of a matrix element-wise.

    :param z: data to apply relu to
    :param keep_prob: the probability of each node to be included in the propagation
    :return:
        a: the result obtained on apply relu activation
        activation_cache: stored value of mask matrix for dropout, and the data passed itself
    """
    assert isinstance(z, np.ndarray) and isinstance(keep_prob, float)
    assert 1. >= keep_prob > 0.

    a = np.maximum(0., z)
    d = np.random.random_sample(size=a.shape)
    d = (d < keep_prob).astype(int)
    a = (a * d) / keep_prob

    activation_cache = {
        "D": d,
        "Z": z,
    }
    assert z.shape == a.shape == d.shape
    return a, activation_cache


def tanh(z, keep_prob):
    """
    Apply tanh activation to all elements of a matrix element-wise.

    :param z: data to apply tanh to
    :param keep_prob: the probability of each node to be included in the propagation
    :return:
        a: the result obtained on apply tanh activation
        activation_cache: stored value of mask matrix for dropout, and the data passed itself
    """
    assert isinstance(z, np.ndarray) and isinstance(keep_prob, float)
    assert 1. >= keep_prob > 0.

    a = np.tanh(z)
    d = np.random.random_sample(size=z.shape)
    d = (d < keep_prob).astype(int)
    a = (a * d) / keep_prob

    activation_cache = {
        "D": d,
        "Z": z,
    }
    assert z.shape == a.shape == d.shape
    return a, activation_cache


def sigmoid_prime(da, activation_cache, keep_prob):
    """
    use the activation function to help select the correct gradients when back propagating

    :param da: the derivative of the cost function with respect to a that is produced in the current layer
    :param activation_cache: the activation cache containing the z that when activated leads to a, and the dropout mask
        matrix
    :param keep_prob: probability to use a node in the layer during one propagation
    :return:
        dZ: the derivative of the cost function wrt z that when activated produces a in the current layer
    """
    assert isinstance(da, np.ndarray) and isinstance(keep_prob, float) and isinstance(activation_cache, dict)
    assert 1. >= keep_prob > 0.
    z, d = activation_cache["Z"], activation_cache["D"]
    assert z.shape == da.shape == d.shape and isinstance(z, np.ndarray) and isinstance(d, np.ndarray)

    da = (da * d) / keep_prob
    a = 1. / (1. + np.exp(-z))
    dz = da * a * (1. - a)
    assert dz.shape == da.shape and isinstance(dz, np.ndarray)
    return dz


def relu_prime(da, activation_cache, keep_prob):
    """
    use the activation function to help select the correct gradients when back propagating

    :param da: the derivative of the cost function with respect to a that is produced in the current layer
    :param activation_cache: the activation cache containing the z that when activated leads to a, and the dropout mask
        matrix
    :param keep_prob: probability to use a node in the layer during one propagation
    :return:
        dZ: the derivative of the cost function wrt z that when activated produces a in the current layer
    """
    assert isinstance(da, np.ndarray) and isinstance(keep_prob, float) and isinstance(activation_cache, dict)
    assert 1. >= keep_prob > 0.
    z, d = activation_cache["Z"], activation_cache["D"]
    assert z.shape == da.shape == d.shape and isinstance(z, np.ndarray) and isinstance(d, np.ndarray)

    da = (da * d) / keep_prob
    dz = np.array(da, copy=True)
    dz[z < 0.] = 0.
    assert dz.shape == da.shape and isinstance(dz, np.ndarray)
    return dz


def tanh_prime(da, activation_cache, keep_prob):
    """
    use the activation function to help select the correct gradients when back propagating

    :param da: the derivative of the cost function with respect to a that is produced in the current layer
    :param activation_cache: the activation cache containing the z that when activated leads to a, and the dropout mask
        matrix
    :param keep_prob: probability to use a node in the layer during one propagation
    :return:
        dZ: the derivative of the cost function wrt z that when activated produces a in the current layer
    """
    assert isinstance(da, np.ndarray) and isinstance(keep_prob, float) and isinstance(activation_cache, dict)
    assert 1. >= keep_prob > 0.
    z, d = activation_cache["Z"], activation_cache["D"]
    assert z.shape == da.shape == d.shape and isinstance(z, np.ndarray) and isinstance(d, np.ndarray)

    da = (da * d) / keep_prob
    a = np.tanh(z)
    dz = da * (1 - np.square(a))
    assert dz.shape == da.shape and isinstance(dz, np.ndarray)
    return dz


def initialize_layer_parameters(dim_prev, dim_now):
    """
    Initialize the Weight and Bias parameter of the layer. Bias to all zeros and Weight to a random uniform
    distribution and scaled by the number of units feeding into the current layer from the previous layer.

    :param dim_prev: the dimensions of the data when coming from the previous layer/number of nodes in the
        previous layer
    :param dim_now: the dimension of the data when it leaves current layer/number of nodes in the current layer
    :return:
        layer_parameters: the parameters, W and b, of the layer that need to be optimized.
    """
    assert isinstance(dim_prev, int) and isinstance(dim_now, int)
    w = np.random.randn(dim_now, dim_prev) * np.sqrt(1. / dim_prev)
    b = np.zeros(shape=(dim_now, 1), dtype=float)
    layer_parameters = {
        "W": w,
        "b": b,
    }
    assert w.shape == (dim_now, dim_prev) and b.shape == (dim_now, 1)
    return layer_parameters


def layer_linear_forward(a_prev, layer_parameters):
    """
    Use the parameters initialized(in the start) and with their updated values(as epochs proceed) along with
    the A parameter being forwarded to the current lauer from the previous layer to obtain the
    z parameter, that needs to be fed to generate the A parameter for the current layer.
    :param a_prev: the A parameter from the previous layer
    :param layer_parameters: parameter dict of Weight and Bias for the current layer
    :return:
        z: the Z parameter for the layer
        linear_cache: cached values to be used during backpropagation
    """
    assert isinstance(a_prev, np.ndarray) and isinstance(layer_parameters, dict)
    w, b = layer_parameters["W"], layer_parameters["b"]
    assert isinstance(w, np.ndarray) and isinstance(b, np.ndarray)
    assert w.shape == (b.shape[0], a_prev.shape[0]) and b.shape[1] == 1
    z = (w @ a_prev) + b
    assert z.shape == (w.shape[0], a_prev.shape[1])
    linear_cache = {
        "A_prev": a_prev,
        "W": w,
        "b": b,
    }
    return z, linear_cache


def layer_linear_activation_forward(a_prev, layer_parameters, keep_prob, layer_activation):
    """
    Obtain the A applying the activation to the Z parameter obtained from the layer_linear_forward function

    :param a_prev: the A parameter from the previous layer
    :param layer_parameters: parameter dict of Weight and Bias for the current layer
    :param keep_prob: the probability of keeping a node during propagation
    :param layer_activation: one of 'relu', 'sigmoid' or 'tanh'
    :return:
        a: the A parameter of the current layer
        cache: the dict containing the linear cache and the activation cache
    """
    assert isinstance(keep_prob, float) and 1. >= keep_prob > 0.
    z, linear_cache = layer_linear_forward(a_prev, layer_parameters)
    if layer_activation == "sigmoid":
        a, activation_cache = sigmoid(z, keep_prob)
    elif layer_activation == "relu":
        a, activation_cache = relu(z, keep_prob)
    else:
        a, activation_cache = tanh(z, keep_prob)
    assert a.shape == z.shape and isinstance(a, np.ndarray)
    assert isinstance(activation_cache, dict) and isinstance(linear_cache, dict)
    layer_cache = {
        "linear": linear_cache,
        "activation": activation_cache,
    }
    return a, layer_cache


def forward_propagation(x, parameters, keep_probs, activations):
    """
    :param x: the inputs to the model
    :param parameters: weights and bias of the layer
    :param keep_probs: probability to randomly use a node
    :param activations: the layer activation; one of 'relu', 'sigmoid' and 'tanh'
    :return:
        yhat: the predictions
        caches: the cached values for all the layers in the model
    """
    assert isinstance(x, np.ndarray) and isinstance(parameters, list)
    assert isinstance(keep_probs, list) and isinstance(activations, list)
    a = np.array(x, copy=True)
    caches = []
    num_layers = len(parameters)
    for layer_num in range(num_layers):
        a_prev = a
        layer_parameters = parameters[layer_num]
        layer_keep_prob = keep_probs[layer_num]
        layer_activation = activations[layer_num]
        a, layer_cache = layer_linear_activation_forward(a_prev, layer_parameters,
                                                         layer_keep_prob, layer_activation)
        caches.append(layer_cache)
    yhat = np.array(a, copy=True)
    assert yhat.shape[1] == x.shape[1]
    return yhat, caches


def compute_cost(yhat, y, lambdas, parameters):
    """
    :param yhat: predictions
    :param y: actual values
    :param lambdas: the regularization parameters for the layers
    :param parameters: weights and biases for the layers
    :return:
        cost: the cost of the wrond predictions
    """
    assert isinstance(yhat, np.ndarray) and isinstance(y, np.ndarray)
    assert isinstance(parameters, list) and isinstance(lambdas, list)
    assert yhat.shape == y.shape and len(lambdas) == len(parameters)
    m = yhat.shape[1]
    cost_cross_entropy = - (1. / m) * np.sum(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
    cost = np.squeeze(cost_cross_entropy)
    num_layers = len(parameters)
    for layer_num in range(num_layers):
        cost += np.squeeze((lambdas[layer_num] / (2. * m)) * np.sum(np.square(parameters[layer_num]["W"])))
    return cost


def layer_linear_backward(dz, linear_cache, layer_lambd):
    """
    Use dZ and the values in the linear_cache and the l2 regularizers to find derivatives wrt to
    A parameter in the previous layer, weight of the current layer and bais of the current layer
    :param dz: derivative wrt Z of the current layer
    :param linear_cache: linear cache to help calculating the gradients
    :param layer_lambd: l2 regularizer for the layer
    :return:
        da_prev, dw, db
    """
    assert isinstance(dz, np.ndarray) and isinstance(linear_cache, dict) and isinstance(layer_lambd, float)
    a_prev, w, b = linear_cache["A_prev"], linear_cache["W"], linear_cache["b"]
    m = a_prev.shape[1]
    dw = (1. / m) * (dz @ a_prev.T) + (layer_lambd / m) * w
    db = (1. / m) * np.sum(dz, axis=1, keepdims=True)
    da_prev = w.T @ dz

    return da_prev, dw, db


def layer_linear_activation_backward(da, layer_cache, layer_keep_prob, layer_activation, layer_lambd):
    """
    Perform back propagation through one layer
    :param da: derivative wrt A in the current layer
    :param layer_cache: linear and activation cache of the current layer
    :param layer_keep_prob: random probs to use a node for the pass
    :param layer_activation: layer activation of the layer
    :param layer_lambd: l2 regularizer for the current layer.
    :return:
        da_prev, dw, db
    """
    assert isinstance(da, np.ndarray) and isinstance(layer_cache, dict)
    assert isinstance(layer_activation, str) and isinstance(layer_keep_prob, float)
    linear_cache, activation_cache = layer_cache["linear"], layer_cache["activation"]
    if layer_activation == "sigmoid":
        dz = sigmoid_prime(da, activation_cache, layer_keep_prob)
    elif layer_activation == "relu":
        dz = relu_prime(da, activation_cache, layer_keep_prob)
    else:
        dz = tanh_prime(da, activation_cache, layer_keep_prob)
    da_prev, dw, db = layer_linear_backward(dz, linear_cache, layer_lambd)
    return da_prev, dw, db


def backward_propagation(yhat, y, parameters, caches, keep_probs, activations, lambdas):
    """
    perform the entire backward propagation through all the layers subsequently calculating all the gradients
    :param yhat: predictions
    :param y: actual values
    :param parameters: weights and biases for the layers
    :param caches: linear and activation cache that help during back propagation
    :param keep_probs: probability to randomly use a node in the layer in one forward and backward propagation
    :param activations: layer activations of each layer
    :param lambdas: l2 regularizers for individual layers
    :return:
        grads: the derivative on the cost function wrt to the weights and bias of all the layers
    """
    assert yhat.shape == y.shape and isinstance(y, np.ndarray) and isinstance(yhat, np.ndarray)
    assert isinstance(parameters, list) and isinstance(keep_probs, list)
    num_layers = len(parameters)
    grads = []
    dyhat = - np.divide(y, yhat) + np.divide(1 - y, 1 - yhat)
    da_prev = np.array(dyhat, copy=True)
    for layer_num in reversed(range(num_layers)):
        layer_cache = caches[layer_num]
        layer_keep_prob = keep_probs[layer_num]
        layer_activation = activations[layer_num]
        layer_lambda = lambdas[layer_num]
        da_prev, dw, db = layer_linear_activation_backward(da_prev, layer_cache, layer_keep_prob,
                                                           layer_activation, layer_lambda)

        layer_grads = {
            "dW": dw,
            "db": db,
        }
        grads.append(layer_grads)
    grads = list(reversed(grads))
    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update the weight and bias parameters from the calculated gradients calculated.
    :param parameters: weights and bias parameters
    :param grads: the gradient of the cost function wrt to the parameters that have to be optimized
    :param learning_rate: the multiplication factor to use for gradient descent
    :return:
        parameters: the updated values of the parameters for the layers when completing one forward and one
                    backward propagation.
    """
    assert isinstance(parameters, list) and isinstance(grads, list) and isinstance(learning_rate, float)
    assert len(parameters) == len(grads)
    for layer_num, _ in enumerate(parameters):
        parameters[layer_num]["W"] -= learning_rate * grads[layer_num]["dW"]
        parameters[layer_num]["b"] -= learning_rate * grads[layer_num]["db"]

    return parameters
