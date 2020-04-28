import numpy as np
from matplotlib import pyplot as plt
from Model.model import *
import os
import h5py


def load_dataset():
    train_file = (str)(os.path.join(os.getcwd(), os.path.join("catvnoncat", "train_catvnoncat.h5")))
    train_dataset = h5py.File(train_file, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_file = (str)(os.path.join(os.getcwd(), os.path.join("catvnoncat", "test_catvnoncat.h5")))
    test_dataset = h5py.File(test_file, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape(1, train_set_y_orig.shape[0])
    test_set_y_orig = test_set_y_orig.reshape(1, test_set_y_orig.shape[0])

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def deep_model(x, y, layers_dims, learning_rate, num_iters, print_cost=True):
    np.random.seed(2)
    costs = []
    parameters = []
    for layer_num in range(1, len(layers_dims)):
        parameters.append(initialize_layer_parameters(layers_dims[layer_num - 1], layers_dims[layer_num]))

    for i in range(num_iters):
        yhat, caches = forward_propagation(x, parameters,
                                           keep_probs=[1., 1., 1., 1.], activations=["relu", "relu", "relu", "sigmoid"])
        cost = compute_cost(yhat, y, lambdas=[0., 0., 0., 0.], parameters=parameters)
        if print_cost and i % 100 == 0:
            print("Cost after", i, "iterations =", cost)
            costs.append(cost)
        grads = backward_propagation(yhat, y, parameters, caches, keep_probs=[1., 1., 1., 1.],
                                     activations=["relu", "relu", "relu", "sigmoid"], lambdas=[0., 0., 0., 0.])
        parameters = update_parameters(parameters, grads, learning_rate=learning_rate)
    plt.plot(costs)
    plt.title("cost function with epochs")
    plt.xlabel("epoch (times 500)")
    plt.ylabel("cost")
    plt.show()
    return parameters


def test(test_x, test_y, parameters):
    yhat = None
    yhat, _ = forward_propagation(test_x, parameters,
                                  keep_probs=[1., 1., 1., 1.], activations=["relu", "relu", "relu", "sigmoid"])
    assert test_y.shape == yhat.shape
    yhat = np.where(yhat < 0.5, 0, 1)
    accuracy = np.mean(yhat == test_y)
    print(accuracy)


def main():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

    # non-cat image
    image_index = 21
    plt.imshow(train_set_x_orig[image_index])
    print("y = " + str(test_set_y[:, image_index]) + ", it is a \'" + str(
        classes[np.squeeze(test_set_y[:, image_index])].decode("utf-8")) + "\' picture")
    plt.show()
    # cat image
    image_index = 42
    plt.imshow(train_set_x_orig[image_index])
    print("y = " + str(test_set_y[:, image_index]) + ", it is a \'" + str(
        classes[np.squeeze(test_set_y[:, image_index])].decode("utf-8")) + "\' picture")
    plt.show()
    # visualize data structure
    print("Train Set X Original Shape:", train_set_x_orig.shape)
    print("Train Set Y Original Shape:", train_set_y.shape)
    print("Test Set X Original Shape", test_set_x_orig.shape)
    print("Test Set Y Original Shape", test_set_y.shape)
    # reshape the data, since we are using a DNN 2D, we need to flatten the images
    train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T / 255
    test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T / 255
    print("Flattened Train Set X:", train_set_x.shape)
    print("Flattened Train Set Y:", train_set_y.shape)
    print("Flattened Test Set X:", test_set_x.shape)
    print("Flattened Test Set Y:", test_set_y.shape)
    # train the model
    parameters = deep_model(train_set_x, train_set_y, layers_dims=[12288, 20, 7, 5, 1], learning_rate=0.0075,
                            num_iters=1000, print_cost=True)
    print("Training accuracy")
    test(train_set_x, train_set_y, parameters)
    print("Testing accuracy")
    test(test_set_x, test_set_y, parameters)


if __name__ == "__main__":
    main()
