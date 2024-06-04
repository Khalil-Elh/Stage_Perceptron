import numpy as np


#This function creat a matrice with a 'w' coefficent for each x parameters the datas have

def initialisation(X):
    W = np.random.randn(X.shape[1],1)
    b = np.random.randn()
    return (W,b)


#This function modelize our decision making model
    #The Z is our linear model
    #The A is the sigmoid function who give us more precition under 0.5 the data is from class 0 over 0.5 is from class 1


def model(X,W,b):
    Z = X.dot(W)+b
    A = 1/(1+np.exp(-Z))
    return A


#This function calculate the precision of the model
    #A is the model prediction
    #Y are the true answers

def cost(A,Y):
    L = -1/len(Y)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    return L

def cost1(A, Y):
    L = 0
    for i in range(len(Y)):
        L += Y[i] * np.log(A[i]) + (1 - Y[i]) * np.log(1 - A[i])
    Lf = -1 / len(Y) * L
    return Lf

#This function calculate the delta who will permit us to update our coefficiant

def gradients(A,X,Y):
    dW = 1/len(Y)*(X.T).dot(A-Y)
    db = 1/len(Y)*np.sum(A-Y)
    return dW,db


#With this function we update the functions

def updatecoeff(W,b,dW,db,alpha):
    W = W-alpha*dW
    b = b-alpha*db
    return W,b

#This function is the loop that train the model it return us out parameter that we can save like that we dosen't have to train the model each time we want to use it
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
yt_and = np.array([[0],[1],[0]])

W,b,Loss = perceptron(data_and,yt_and,0.5)

x1=np.array([[0,0]])

print(prediction(x1,W,b))
