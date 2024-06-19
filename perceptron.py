import numpy as np
import matplotlib.pyplot as plt
import operator


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
    epsilon=1e-15
    L = (-1/len(Y))*np.sum(Y*np.log(A+epsilon)+(1-Y)*np.log(1-A+epsilon))
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
    itteration=int(input("Entrez un nombre d'itteration: "))
    W,b = initialisation(X)
    Loss=[]
    for i in range (itteration):
        A = model(X,W,b)
        current_loss=cost(A,Y)
        Loss.append(current_loss)
        dW,db = gradients(A,X,Y)
        W,b = updatecoeff(W,b,dW,db,alpha)
        if i%100==0:
            print(f"Itteration {i}: Loss {current_loss}")
    return W,b,Loss


def prediction(X,W,b):
    A=model(X,W,b)
    return A >= 0.5


#data_and = np.array([[1,0],[1,1],[1,0]])
#yt_and = np.array([[0],[1],[0]])

#W,b,Loss = perceptron(data_and,yt_and,0.5)

#x1=np.array([[0,0]])

#print(prediction(x1,W,b))


#faire la fonction pour générer les datas!
#penser à tuner le pas

def createDataSet(n,m,op):
    X=np.random.randint(0,2,size=(m,n))
    Y=op(X[:,0],X[:,1]).reshape(-1,1)
    split_index=m*3//4
    Xt=X[:split_index]
    Xv=X[split_index:]
    Yt=Y[:split_index]
    Yv=Y[split_index:]
    return Xt,Xv,Yt,Yv

def creatgraph(L,name,file_name,step):
    fig, ax = plt.subplots()
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f')) 
    for i in range (len(L)):
        ax.plot(L[i],label=f"step : {step[i]}")
    ax.legend()
    ax.set_xlabel("Number of itteration")
    ax.set_ylabel("Loss function")
    ax.set_title(name)
    fig.savefig("/Users/khalil_elhajoui/Documents/Projet Perso/Stage/L1/Stage_Perceptron/Graph/"+file_name)
    plt.close(fig) 


def testpercep(test_number,op):
    m = int(input("Enter how many data points you wanna create: "))
    n = 2
    Xt,Xv,Yt,Yv = createDataSet(n,m,op)
    l_loss=[]
    l_step=[]
    for i in range (test_number):
        step = float(input("Enter a training step: "))
        W,b,loss = perceptron(Xt,Yt,step)
        l_loss.append(loss)
        l_step.append(str(step))
    creatgraph(l_loss,f"Graph of Log loss "+str(op),str(op)+f"loss_graph.png",l_step)
    return Xt, Xv, Yt, Yv

#testpercep(5,operator.and_)

def lineardecisionline(op):
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y = Y=op(X[:,0],X[:,1]).reshape(-1,1)
    w,b,loss = perceptron(X,Y,0.03)
    xx, yy = np.meshgrid(np.linspace(-0.5,1.5,200), np.linspace(-0.5, 1.5, 200))
    grid = np.c_[xx.ravel(),yy.ravel()]
    probs = model(grid, w, b).reshape(xx.shape)
    fig, ax = plt.subplots()
    ax.scatter(X[:,0], X[:,1],c=Y.flatten(), cmap= plt.cm.bwr, edgecolor='k')
    ax.contour(xx, yy, probs, levels=[0.5], cmap="Greys_r")
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlim(-0.5, 1.5)
    ax.set_title("Decision line for:"+str(op))
    fig.savefig("/Users/khalil_elhajoui/Documents/Projet Perso/Stage/L1/Stage_Perceptron/Graph/Decision line for "+str(op)+".png")
    plt.close(fig)

lineardecisionline(operator.xor)

