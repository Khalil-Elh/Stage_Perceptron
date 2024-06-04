import numpy as np

def initialisation(X):
    W = np.random.randn(X.shape[1],1)
    b = np.random.randn(1)
    return (W,b)

def model(X,W,b):
    Z = X.dot(W)+b
    A = 1/(1+np.exp(-Z))
    return A

def cost(A,Y):
    L = -1/(Y.shape[0])*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    return L

def gradients(A,X,Y):
    dW = 1/Y.shape[0]*(X.T).dot(A-Y)
    db = 1/Y.shape[0]*np.sum(A-Y)
    return dW,db

def updatecoeff(W,b,dW,db,alpha):
    W = W-alpha*dW
    b = b-alpha*db
    return W,b


def perceptron(X,Y,alpha):
    seuil=int(input("Entrez un nombre d'itteration"))
    W,b = initialisation(X)
    A = model(X,W,b)
    Loss=[]
    for i in range (seuil):
        A = model(X,W,b)
        Loss.append(cost(A,Y))
        dW,db = gradients(A,X,Y)
        W,b = updatecoeff(W,b,dW,db,alpha)
        print(Loss[-1])
    return W,b,Loss


def prediction(X,W,b):
    A=model(X,W,b)
    return A >= 0.5


data_and = np.array([[1,0],[1,1],[1,0]])
yt_and = np.array([0,1,0])

W,b,Loss = perceptron(data_and,yt_and,0.5)

x1=np.array([[1,1]])

print(prediction(x1,W,b))