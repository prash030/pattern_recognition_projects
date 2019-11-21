# intro.py (c) Prasanth Ganesan
# Author: Prash Ganesan <prasganesan.pg@gmail.com>
# Simple pattern recognition algorithms

# This section contains a python program to perform classification of two classes using least squares method
from pylab import *

# Function to calculate parameters for least squares
def lsq(X,Xtest, Y):
    # Calculate beta
    beta = np.dot(np.linalg.pinv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X), Y))

    # Classification
    [k1,k2] = [0,0]
    Ctrain=list([])
    for i in range(0,len(Xtest)):
        C = np.dot(beta,np.transpose(Xtest[i,:]))
        if C > 0.5:
            if k1 == 0:
                C1 = Xtest[i,-2:]
                Ctrain.append(1)
            else:
                C1 = np.vstack((C1,Xtest[i,-2:]))
                Ctrain.append(1)

            k1 = 1

        else:
            if k2 == 0:
                C2 = Xtest[i,-2:]
                Ctrain.append(0)
            else:
                C2 = np.vstack((C2,Xtest[i,-2:]))
                Ctrain.append(0)

            k2 = 1
    C1=array(C1)
    C2=array(C2)
    Ctrain=array(Ctrain)
    return [C1,C2,Ctrain, beta]

# Function to apply the lsq for all the grid points
def lsqgrid(data,X,Y,res):
    xgrid=[]
    ygrid=[]
    for row in range(int(min(data[:,0])-1)*res,int(max(data[:,0])+1)*res,1):
        for col in range(int(min(data[:,1])-1)*res,int(max(data[:,1])+1)*res,1):
            xgrid.append(row/res)
            ygrid.append(col/res)
    xgrid=array(xgrid)
    ygrid=array(ygrid)

    # Testing
    X1test = np.ones(len(xgrid))
    X1test = np.expand_dims(X1test,axis=1)
    X1grid = np.expand_dims(xgrid,axis=1)
    Y1grid = np.expand_dims(ygrid,axis=1)
    Xtest = np.hstack((X1test,np.hstack((X1grid,Y1grid))))
    [C1,C2,_,beta] = lsq(X,Xtest,Y)

    return[Xtest,C1,C2,beta]

# MAIN
data = load("data.npy")
resolution = 100 # increase or decrease the resolution of grid points here
X = data[0,:-1]
X1 = np.ones(len(data))
X1 = np.expand_dims(X1,axis=1)
for i in range(1,len(data)):
    X = np.vstack((X,data[i,:-1]))

X = np.hstack((X1,X))
Y = data[:,2]

[Xtest,C1,C2,beta] = lsqgrid(data,X,Y,resolution)

linea = (-1*(beta[1]/beta[2]) * min(Xtest[:,1])) + ((0.5 - beta[0])/beta[2])
lineb = (-1*(beta[1]/beta[2]) * max(Xtest[:,1])) + ((0.5 - beta[0])/beta[2])

# Plot the regions, line and the points on scatter plot
plt.scatter(C1[:,0], C1[:,1], c='green', edgecolor="green")
plt.scatter(C2[:,0], C2[:,1],c='yellow', edgecolor="yellow")
plt.plot([min(Xtest[:,1]),max(Xtest[:,1])], [linea,lineb],linewidth=4.5, c="black" )
for i in range(0,len(data)):
    if data[i,2]==1:
        plt.scatter(data[i,0], data[i,1],s=50, c="orange")
    else:
        plt.scatter(data[i,0], data[i,1],s=50, c="blue")
plt.show()

# TO compute confusion matrix
[_,_,Ctrain,_] = lsq(X,X,Y)
Confusion1 = Ctrain + Y
Confusion2 = Ctrain - Y
[count0,count2,count1,countm1]=[int(0),int(0),int(0),int(0)]
for i in range(len(Confusion1)):
    if Confusion1[i] == 0:
        count0+=1
    elif Confusion1[i] == 2:
        count2+=1
    if Confusion2[i] == 1:
        count1+=1
    elif Confusion2[i] == -1:
        countm1+=1
Confmat = array([[count0,countm1],[count1,count2]])
print(Confmat)

# Percentage of correct classification
percentage = ((Confmat[0,0] + Confmat[1,1])/len(data))*100
print(percentage)

##############################################################################################################

# This section is the program to perform k-nearest neighbors classification

from pylab import *

# Function to calculate parameters for knn
def knn(X,X1,Y,K1):
    # Eucledean distance
    Class = []
    for i in range(len(X)):
        dist=[]
        for j in range(len(X1)):
            dist.append(math.sqrt(pow((X1[j][0] - X[i][0]), 2) + pow((X1[j][1] - X[i][1]), 2)))
        dist = array(dist)

    # Finding K mins
        nn = []
        [K, offset] = [K1, 100]
        while K > 0:
            m = min(dist)
            current = np.where(dist == m)
            current = array(current)
            for temp in range(len(current)):
                nn.append(current[temp])
                dist[current[temp]] = max(dist) + offset
            K -=1

        nn = array(nn)

        # Classification
        Classtemp = []
        for temp in range(len(nn)):
            Classtemp.append(Y[nn[temp]])
        Classtemp = array(Classtemp)
        if sum(Classtemp)/len(Classtemp) > 0.5:
            Class.append(1)
        else:
            Class.append(0)

    Class = array(Class)
    return Class

# Train the input points using knn
def knntraining(data,K):
    # Parse the input and output
    X = data[0,:-1]
    for i in range(1,len(data)):
        X = np.vstack((X,data[i,:-1]))

    Y = data[:,2]
    # Call the function
    Class = knn(X,X,Y,K)
    return [X,Class]

# Function to apply the knn for all the grid points
def knngrid(data,traverserow,traversecol,res,K):
    xgrid=[]
    ygrid=[]
    if traverserow == 1:
        for col in range(int(min(data[:,1])-1)*res,int(max(data[:,1])+1)*res,1):
            for row in range(int(min(data[:,0])-1)*res,int(max(data[:,0])+1)*res,1):
                xgrid.append(row/res)
                ygrid.append(col/res)
    elif traversecol==1:
        for row in range(int(min(data[:,0])-1)*res,int(max(data[:,0])+1)*res,1):
            for col in range(int(min(data[:,1])-1)*res,int(max(data[:,1])+1)*res,1):
                xgrid.append(row/res)
                ygrid.append(col/res)
    else:
        print("The traverse flags are wrong. One should be zero and the other should be 1")
        return 0
    xgrid=array(xgrid)
    ygrid=array(ygrid)
    X = np.vstack((xgrid,ygrid))
    X = np.transpose(X)
    [X1,Y] = knntraining(data,K)
    Class = knn(X,X1,Y,K)

    # Store different classes
    C1 = array([])
    C2 = array([])
    for i in range(len(X)):
        if Class[i] == 1:
            if C1.size==0:
                C1 = X[i,:]
            else:
                C1 = np.vstack((C1,X[i,:]))
        else:
            if C2.size==0:
                C2 = X[i,:]
            else:
                C2 = np.vstack((C2,X[i,:]))

    k12 = 0
    for i in range(len(X)-1):
        if (Class[i+1]-Class[i] != 0) and (X[i,0] != max(xgrid)) and (X[i,1] != max(ygrid)):
            if k12==0:
                Cbound = X[i,:]
            else:
                Cbound = np.vstack((Cbound,X[i,:]))
            k12 = 1
    return[C1,C2,Cbound]

# MAIN
data = load("data.npy")
K = 15 # change this to 1 for 1 nearest neighbor
resolution = 100 # increase or decrease the resolution of grid points here
[C1,C2,Cbound] = knngrid(data,0,1,resolution,K)
[_,_,Cbound2] = knngrid(data,1,0,resolution,K)

# Produce the plots
plt.scatter(C1[:,0], C1[:,1], c='green',edgecolor="green")
plt.scatter(C2[:,0], C2[:,1], c='yellow',edgecolor="yellow")
plt.scatter(Cbound[:,0], Cbound[:,1], c='black')
plt.scatter(Cbound2[:,0], Cbound2[:,1], c='black')
for i in range(0,len(data)):
        if data[i,2]==1:
            plt.scatter(data[i,0], data[i,1], s=50, c="orange")
        else:
            plt.scatter(data[i,0], data[i,1], s=50, c="blue")
plt.show()

# TO compute confusion matrix
[_,Ctrain] = knntraining(data,K)
Confusion1 = Ctrain + data[:,2]
Confusion2 = Ctrain - data[:,2]
#print(Ctrain)
#print(data[:,2])
[count0,count2,count1,countm1]=[int(0),int(0),int(0),int(0)]
for i in range(len(Confusion1)):
    if Confusion1[i] == 0:
        count0+=1
    elif Confusion1[i] == 2:
        count2+=1
    if Confusion2[i] == 1:
        count1+=1
    elif Confusion2[i] == -1:
        countm1+=1
Confmat = array([[count0,countm1],[count1,count2]])
print(Confmat)

# Percentage of correct classification
percentage = ((Confmat[0,0] + Confmat[1,1])/len(data))*100
print(percentage)

