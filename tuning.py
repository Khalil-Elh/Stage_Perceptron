import math
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, accuracy_score
import operator

def activate(Z, activation_f):

    if activation_f == 'sigmoid':
        return 1 / (1 + np.exp(-Z))
    elif activation_f == 'relu':
        return np.maximum(0, Z)
    elif activation_f == 'tanh':
        return np.tanh(Z)
    else:
        raise ValueError("Unsupported activation function")

def activate_derivative(Z, activation_f):

    if activation_f == 'sigmoid':
        A = activate(Z, 'sigmoid')
        return A * (1 - A)
    elif activation_f == 'relu':
        return np.where(Z > 0, 1, 0)
    elif activation_f == 'tanh':
        A = activate(Z, 'tanh')
        return 1 - np.square(A)
    else:
        raise ValueError("Unsupported activation function")


def initialisation(layer_dims):

    np.random.seed(42)
    params = {}
    L = len(layer_dims)

    for l in range(1, L):
        params[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        params[f'b{l}'] = np.zeros((layer_dims[l], 1))

    return params


def forward_propagation(X, params, activation_f='sigmoid'):

    caches = []
    A = X
    L = len(params) // 2

    for l in range(1, L + 1):
        W = params[f'W{l}']
        b = params[f'b{l}']
        Z = np.dot(W, A) + b
        A = activate(Z, activation_f)
        caches.append((A, W, b, Z))

    return A, caches

def back_propagation(X, Y, params, caches, activation_f='sigmoid'):

    grads = {}
    L = len(caches)
    m = X.shape[1]
    Y = Y.reshape(caches[-1][0].shape)

    dZ = caches[-1][0] - Y
    grads[f'dW{L}'] = 1/m * np.dot(dZ, caches[-2][0].T)
    grads[f'db{L}'] = 1/m * np.sum(dZ, axis=1, keepdims=True)

    for l in reversed(range(1, L)):
        A_prev, W, b, Z = caches[l-1]
        dZ_prev = np.dot(params[f'W{l+1}'].T, dZ) * activate_derivative(Z, activation_f)
        grads[f'dW{l}'] = 1/m * np.dot(dZ_prev, A_prev.T)
        grads[f'db{l}'] = 1/m * np.sum(dZ_prev, axis=1, keepdims=True)
        dZ = dZ_prev

    return grads


def update(params, grads, learning_rate):
    
    L = len(params) // 2

    for l in range(1, L + 1):
        params[f'W{l}'] -= learning_rate * grads[f'dW{l}']
        params[f'b{l}'] -= learning_rate * grads[f'db{l}']

    return params



def prediction(X, params, activation_f='sigmoid'):

    A, _ = forward_propagation(X, params, activation_f)

    return A >= 0.5

def neural_network(X_train, Y_train, layer_dims, learning_rate=0.3, n_iter=10000, activation_f='sigmoid'):

    params = initialisation(layer_dims)
    train_loss = []
    train_acc = []

    for i in range(n_iter):
        A, caches = forward_propagation(X_train, params, activation_f)
        grads = back_propagation(X_train, Y_train, params, caches, activation_f)
        params = update(params, grads, learning_rate)

        if i % 10 == 0:
            train_loss.append(log_loss(Y_train.flatten(), A.flatten()))
            Y_pred = prediction(X_train, params, activation_f)
            cur_accuracy = accuracy_score(Y_train.flatten(), Y_pred.flatten())
            train_acc.append(cur_accuracy)

    return params, train_loss, train_acc


def createDataSet(n, m, op):

    X = np.random.randint(0, 2, size=(m, n))
    Y = op(X[:, 0], X[:, 1]).reshape(-1, 1)
    split_index = m * 3 // 4
    Xt = X[:split_index].T
    Xv = X[split_index:].T
    Yt = Y[:split_index].T
    Yv = Y[split_index:].T

    return Xt, Xv, Yt, Yv


def creatgraph(L, name, file_name, step):
    fig, ax = plt.subplots()
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    for i in range(len(L)):
        ax.plot(L[i], label=f"step: {step[i]}")
    ax.legend()
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Loss function")
    ax.set_title(name)
    fig.savefig("/Users/khalil_elhajoui/Documents/Projet Perso/Stage/L1/Stage_Perceptron/Graph/"+file_name)
    plt.close(fig)
    return

def test_neural_net(op, l_rate=[0.09, 0.07, 0.05, 0.03, 0.01]):

    m = int(input("Enter how many data points you wanna create: "))
    n = 2
    Xt, Xv, Yt, Yv = createDataSet(n, m, op)
    l_loss = []
    l_step = []
    for i, lr in enumerate(l_rate):
        params, loss, accuracy = neural_network(Xt, Yt, [2, 1], learning_rate=lr)
        l_loss.append(loss)
        l_step.append(str(lr))
    creatgraph(l_loss, f"Graph of Log loss {str(op)}", f"{str(op)}_loss_graph_neural_net.png", l_step)
    return Xt, Xv, Yt, Yv

def decisionline(op, layer_dim=[2,1]):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = op(X[:, 0], X[:, 1]).reshape(-1, 1).T

    params, _, _ = neural_network(X.T, Y, layer_dims=layer_dim)
    
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    grid = np.c_[xx.ravel(), yy.ravel()].T
    probs = forward_propagation(grid, params)['A2'].reshape(xx.shape)
    
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=Y.flatten(), cmap=plt.cm.bwr, edgecolor='k')
    ax.contour(xx, yy, probs, levels=[0.5], cmap="Greys_r")
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlim(-0.5, 1.5)
    ax.set_title(f"Decision line for: {str(op)}")
    fig.savefig("/Users/khalil_elhajoui/Documents/Projet Perso/Stage/L1/Stage_Perceptron/Graph/Neural_net decision line for "+str(op)+".png")
    plt.close(fig)

#test_neural_net(5, operator.or_)
#decisionline(operator.or_)


def tune_model(layer_dims_list, learning_rates, activation_functions, op):

    best_loss = math.inf
    best_params = {}
    Xt, Xv, Yt, Yv = createDataSet(layer_dims_list[0][0], 2, op)
    
    num_activations = len(activation_functions)
    num_lr = len(learning_rates)
    num_layer_dims = len(layer_dims_list)
    
    activation_loss = np.zeros((num_activations, num_lr, num_layer_dims))

    for i, activation_f in enumerate(activation_functions):
        for j, learning_rate in enumerate(learning_rates):
            for k, layer_dims in enumerate(layer_dims_list):
                params, train_loss, train_acc = neural_network(Xt, Yt, layer_dims, learning_rate=learning_rate, n_iter=10000, activation_f=activation_f)
                A_val, _ = forward_propagation(Xv, params, activation_f)
                val_loss = log_loss(Yv.flatten(), A_val.flatten())

                activation_loss[i, j, k] = val_loss
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = {
                        'layer_dims': layer_dims,
                        'activation_f': activation_f,
                        'learning_rate': learning_rate,
                        'params': params
                    }

    return best_params, best_loss, activation_loss

layer_dims_list = [
    [2, 4, 1],
    [2, 5, 3, 1],
    [2, 6, 4, 2, 1]
]
learning_rates = [0.09, 0.07, 0.05, 0.03, 0.01]
activation_functions = ['sigmoid', 'relu', 'tanh']


best_params, best_loss, activation_loss = tune_model(layer_dims_list, learning_rates, activation_functions, operator.or_)
print(f"Best Layer Dimensions: {best_params['layer_dims']}")
print(f"Best Activation Function: {best_params['activation_f']}")
print(f"Best Learning Rate: {best_params['learning_rate']}")
print(f"Best Loss: {best_loss}")
print(f"Activation Loss Matrix:\n {activation_loss}")