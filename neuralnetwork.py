import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, accuracy_score
import operator

def initialisation(n0, n1, n2):
    W1 = np.random.randn(n1,n0)
    b1 = np.random.randn(n1,1)
    W2 = np.random.randn(n2,n1)
    b2 = np.random.randn(n2,1)

    params = { 'W1' : W1,
               'b1' : b1,
               'W2' : W2,
               'b2' : b2
             }

    return params


def forward_propagation(X, params):

    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    Z1 = W1.dot(X)+b1
    A1 = 1 / (1+np.exp(-Z1))
    Z2 = W2.dot(A1)+b2
    A2 = 1 / (1+np.exp(-Z2))
    
    activations = { 'A1' : A1,
                    'A2' : A2
                   }

    return activations


def back_propagation(X, Y, params, activations):

    A1 = activations['A1']
    A2 = activations['A2']
    W2 = params['W2']

    m=Y.shape[1]

    dZ2 = A2-Y
    dZ1 = (W2.T).dot(dZ2) * A1 * (1-A1)

    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    grads = { 'dW1' : dW1,
              'db1' : db1,
              'dW2' : dW2,
              'db2' : db2
             }

    return grads


def update(params, grads, learning_rate):

   params['W1'] -= learning_rate * grads['dW1']
   params['b1'] -= learning_rate * grads['db1']
   params['W2'] -= learning_rate * grads['dW2']
   params['b2'] -= learning_rate * grads['db2']

   return params

def prediction(X, params):
    
    A = forward_propagation(X, params)

    return A['A2'] >= 0.5


def neural_network(X_train, Y_train, n1 = 3, learning_rate = 0.3, n_iter = 10000):

    n0 = X_train.shape[0]
    n2 = Y_train.shape[0]
    params = initialisation(n0, n1, n2)

    train_loss = []
    train_acc = []

    for i in range (n_iter):
        
        activations = forward_propagation(X_train, params)
        grads = back_propagation(X_train, Y_train, params, activations)
        params = update(params, grads, learning_rate)

        if i % 10 == 0:
            train_loss.append(log_loss(Y_train.flatten(), activations['A2'].flatten()))
            Y_pred = prediction(X_train, params)
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

def test_neural_net(test_number, op):
    m = int(input("Enter how many data points you wanna create: "))
    n = 2
    Xt, Xv, Yt, Yv = createDataSet(n, m, op)
    l_loss = []
    l_step = []
    for i in range(test_number):
        step = float(input("Enter a training step: "))
        params, loss, accuracy = neural_network(Xt, Yt, learning_rate=step)
        l_loss.append(loss)
        l_step.append(str(step))
    creatgraph(l_loss, f"Graph of Log loss {str(op)}", f"{str(op)}_loss_graph_neural_net.png", l_step)
    return Xt, Xv, Yt, Yv

def decisionline(op):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = op(X[:, 0], X[:, 1]).reshape(-1, 1).T
    params, loss, accuracy = neural_network(X.T, Y)
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

test_neural_net(5, operator.or_)
decisionline(operator.or_)