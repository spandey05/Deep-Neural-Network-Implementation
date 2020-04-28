from Model.model import *
import numpy as np
from matplotlib import pyplot as plt


def deep_model(x, y, layers_dims, learning_rate, num_iters, print_cost=True):
    np.random.seed(2)
    costs = []
    parameters = []
    for layer_num in range(1, len(layers_dims)):
        parameters.append(initialize_layer_parameters(layers_dims[layer_num - 1], layers_dims[layer_num]))

    for i in range(num_iters):
        yhat, caches = forward_propagation(x, parameters,
                                           keep_probs=[1., 1., 1.], activations=["tanh", "tanh", "sigmoid"])
        cost = compute_cost(yhat, Y, lambdas=[0.1, 0., 0.], parameters=parameters)
        if print_cost and i % 500 == 0:
            print("Cost after", i, "iterations =", cost)
            costs.append(cost)
        grads = backward_propagation(yhat, y, parameters, caches, keep_probs=[1., 1., 1.],
                                     activations=["tanh", "tanh", "sigmoid"], lambdas=[0.1, 0., 0.])
        parameters = update_parameters(parameters, grads, learning_rate=learning_rate)
    print(y)
    print(yhat)
    plt.plot(costs)
    plt.title("cost function with epochs")
    plt.xlabel("epoch (times 500)")
    plt.ylabel("cost")
    plt.show()


if __name__ == "__main__":
    X = np.array([[0., 1., 0., 1.],
                  [0., 0., 1., 1.]], dtype=float)
    Y = np.array([[0., 1., 1., 0.]], dtype=float)
    deep_model(X, Y, layers_dims=[2, 4, 4, 1],
               learning_rate=0.1, num_iters=30000, print_cost=True)
