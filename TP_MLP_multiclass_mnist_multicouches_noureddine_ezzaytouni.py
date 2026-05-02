# TP_MLP_multiclass_mnist_multicouches.py

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

np.random.seed(42)

# ====================== 1. INITIALISATION ======================
def initialize_parameters(layers_dims):
    parameters = {}
    L = len(layers_dims) - 1

    for l in range(1, L + 1):
        n_prev = layers_dims[l-1]
        n_current = layers_dims[l]

        parameters[f"W{l}"] = np.random.randn(n_current, n_prev) * 0.01
        parameters[f"b{l}"] = np.zeros((n_current, 1))

        assert parameters[f"W{l}"].shape == (n_current, n_prev)
        assert parameters[f"b{l}"].shape == (n_current, 1)

    return parameters


# ====================== 2. ACTIVATIONS ======================
def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return Z > 0

def softmax(Z):
    Z_shift = Z - np.max(Z, axis=0, keepdims=True)  # stabilité numérique
    exp_Z = np.exp(Z_shift)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


# ====================== 3. FORWARD PROPAGATION ======================
def forward_propagation(X, parameters):
    cache = {}
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        W = parameters[f"W{l}"]
        b = parameters[f"b{l}"]

        Z = np.dot(W, A) + b
        A = relu(Z)

        cache[f"Z{l}"] = Z
        cache[f"A{l}"] = A

    # Dernière couche (softmax)
    WL = parameters[f"W{L}"]
    bL = parameters[f"b{L}"]

    ZL = np.dot(WL, A) + bL
    AL = softmax(ZL)

    cache[f"Z{L}"] = ZL
    cache[f"A{L}"] = AL

    return AL, cache


# ====================== 4. COUT ======================
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -np.sum(Y * np.log(AL + 1e-8)) / m
    return np.squeeze(cost)


# ====================== 5. BACKPROPAGATION ======================
def backward_propagation(X, Y, cache, parameters):
    grads = {}
    L = len(parameters) // 2
    m = X.shape[1]

    AL = cache[f"A{L}"]

    # Dernière couche
    dZ = AL - Y

    for l in reversed(range(1, L + 1)):
        A_prev = X if l == 1 else cache[f"A{l-1}"]
        W = parameters[f"W{l}"]

        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)

        grads[f"dW{l}"] = dW
        grads[f"db{l}"] = db

        if l > 1:
            dA_prev = np.dot(W.T, dZ)
            Z_prev = cache[f"Z{l-1}"]
            dZ = dA_prev * relu_derivative(Z_prev)

    return grads


# ====================== 6. UPDATE ======================
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
        parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]

    return parameters


# ====================== 7. MODELE ======================
def model(X, Y, layers_dims, num_iterations=2000, learning_rate=0.1, print_cost=False):
    costs = []
    parameters = initialize_parameters(layers_dims)

    for i in range(num_iterations):
        AL, cache = forward_propagation(X, parameters)
        cost = compute_cost(AL, Y)
        grads = backward_propagation(X, Y, cache, parameters)
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 500 == 0:
            print(f"Itération {i:5d} | Coût = {cost:.6f}")

    return parameters, costs


# ====================== FONCTIONS AUXILIAIRES ======================
def predict(X, parameters):
    AL, _ = forward_propagation(X, parameters)
    return np.argmax(AL, axis=0)

def one_hot(Y, num_classes=10):
    return np.eye(num_classes)[Y].T


# ====================== MAIN ======================
if __name__ == "__main__":

    # Chargement MNIST
    (X_train_raw, Y_train_raw), (X_test_raw, Y_test_raw) = mnist.load_data()

    # Prétraitement
    X_train = X_train_raw.reshape(X_train_raw.shape[0], -1).T / 255.0
    X_test = X_test_raw.reshape(X_test_raw.shape[0], -1).T / 255.0

    Y_train = one_hot(Y_train_raw)
    Y_test = one_hot(Y_test_raw)

    print(f"X_train.shape = {X_train.shape}")
    print(f"Y_train.shape = {Y_train.shape}")
    print(f"X_test.shape = {X_test.shape}")

    # Architectures à tester
    architectures = [
        [784, 128, 10],
        [784, 256, 128, 10],
        [784, 512, 256, 128, 10]
    ]

    plt.figure()

    for arch in architectures:
        print(f"\n===== Architecture {arch} =====")

        parameters, costs = model(
            X_train, Y_train,
            layers_dims=arch,
            num_iterations=2000,
            learning_rate=0.1,
            print_cost=True
        )

        preds = predict(X_test, parameters)
        labels = np.argmax(Y_test, axis=0)

        accuracy = np.mean(preds == labels)
        print(f"Accuracy: {accuracy * 100:.2f}%")

        plt.plot(costs, label=str(arch))

    plt.title("Evolution du coût")
    plt.xlabel("Itérations (x100)")
    plt.ylabel("Coût")
    plt.legend()
    plt.show()
