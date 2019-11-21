# Program to classify the classes in data using Bayes classifier

from pylab import *

# Function for counting number of classes to use in finding prior
def count(C,data):
    num=0
    for i in range(len(data)):
        if data[i,-1] == C:
            num+=1
    return num

# Bayesian and MAP classification function
def bayesian_map(data, griddata, method): # Assumes D=2; method = 1 for MAP, 2 for Bayesian with uniform cost and 3 for bayesian with given cost matrix
    data = array(data)
    Y = array(data[:,2])
    Xgrid = griddata # should have two columns

    # Find the prior
    Classes = np.unique(Y)

    C = []
    for i in range(len(Classes)):
        C.append(count(Classes[i],data))

    prior = []
    for i in range(len(Classes)):
        prior.append(C[i]/len(data))

    if sum(prior) != 1:
        print("Error: Prior doesn't sum to 1, check your computation")
        return 0

    # covariance matrix of each class
    for i in range(len(Classes)):
        row = np.where(Y == Classes[i])
        row = array(row)
        row = np.transpose(row)
        for j in range(len(row)):
            if j==0:
                X = data[row[j],:-1]
            else:
                X = np.vstack((X,data[row[j],:-1]))
        mu_x = sum(X[:,0])/len(X)
        mu_y = sum(X[:,1])/len(X)
        var_x = sum(np.square(X[:,0]-mu_x))/(len(X)-1)
        var_y = sum(np.square(X[:,1]-mu_y))/(len(X)-1)
        covar_xy = sum((array(X[:,0])-mu_x) * (array(X[:,1])-mu_y))/(len(X)-1)
        covmatrix = array([[var_x, covar_xy], [covar_xy,var_y]])
        mu = np.hstack((mu_x,mu_y))
        D = data.shape[1] - 1

        # class conditionals
        if i==0:
            N = np.exp(-0.5*(np.dot(np.dot((Xgrid-mu),(np.linalg.pinv(covmatrix))),(np.transpose(Xgrid-mu)))))/((pow((2*np.pi),(D/2)))*(pow(np.linalg.det(covmatrix),0.5)))
            N = np.diag(N)
        else:
            N0 = np.exp(-0.5*(np.dot(np.dot((Xgrid-mu),(np.linalg.pinv(covmatrix))),(np.transpose(Xgrid-mu)))))/((pow((2*3.14),(D/2)))*(pow(np.linalg.det(covmatrix),0.5)))
            N = np.vstack((N,np.diag(N0)))
    N = np.transpose(N)

    # Do computations for bayesian and MAP
    if method == 1:
        print("MAP classification")
        for i in range(N.shape[1]):
            if i==0:
                post = N[:,i]*prior[i]
            else:
                post = np.vstack((post,N[:,i]*prior[i]))
        post = np.transpose(post)

        # Now classify according to MAP
        label = []
        for i in range(len(Xgrid)):
            if post[i,0] > post[i,1]:
                label.append(0)
            else:
                label.append(1)
        label = array(label)
    elif method == 2:
        print("Bayesain with uniform cost")
        cost_uni = 1 - np.identity(len(Classes))
        Risk = np.dot(N,cost_uni)
        for i in range(Risk.shape[1]):
            if i==0:
                post = Risk[:,i]*prior[i]
            else:
                post = np.vstack((post,Risk[:,i]*prior[i]))
        post = np.transpose(post)

        # Now classify according to bayesian min risk
        label = []
        for i in range(len(Xgrid)):
            temp = np.where(post[i,:] == min(post[i,:]))
            label.append(temp[0]+1)
        label = array(label)

    elif method == 3:
        print("Bayesain with cost matrix")
        print("Bayesain with non-uniform cost matrix and increased prior")
        cost_nonuni = array([[-0.2,0.07,0.07,0.07],[0.07,-0.15,0.07,0.07],[0.07,0.07,-0.05,0.07],[0.03,0.03,0.03,0.03]])
        Risk = np.dot(N,cost_nonuni)
        for i in range(Risk.shape[1]):
            if i==0:
                post = Risk[:,i]*prior[i]
                #print(post)
            else:
                post = np.vstack((post,Risk[:,i]*prior[i]))
        post = np.transpose(post)

        # Now classify according to bayesian min risk
        label = []
        for i in range(len(Xgrid)):
            temp = np.where(post[i,:] == min(post[i,:]))
            label.append(temp[0]+1)
        label = array(label)
    else:
        print("Method has to be 1, 2 or 3")
        return 0
    return label

# Function to create grid and output the class labels for regions
def gridpts(data, traverserow,traversecol,res,method):
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
    griddata = np.transpose(np.vstack((xgrid,ygrid)))

    label = bayesian_map(data, griddata, method)
    # Find the boundaries
    k12 = 0
    for i in range(len(griddata)-1):
        if (label[i+1]-label[i] != 0) and (griddata[i,0] != max(xgrid)) and (griddata[i,1] != max(ygrid)):
            if k12==0:
                Cbound = griddata[i,:]
            else:
                Cbound = np.vstack((Cbound,griddata[i,:]))
            k12 = 1
    return(griddata,label,Cbound)

# MAIN
# import npy data
data = load("data.npy")

# Training
label = bayesian_map(data, data[:,:-1], 1)

# Make confusion matrix
Confusion1 = label + data[:,2]
Confusion2 = label - data[:,2]
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
print('Confusion Matrix:')
print(Confmat)

# Percentage of correct classification
percentage = ((Confmat[0,0] + Confmat[1,1])/len(data))*100
print('Percentage of correct classification:')
print(percentage)

# Testing - Apply the classifier for all the grid points
resolution = 10 # Change the resolution here
[griddata,label,Cbound] = gridpts(data,0,1,resolution,1)
[_,_,Cbound2] = gridpts(data,1,0,resolution,1)

# Plot the data, regions and the classification boundaries
for i in range(len(griddata)):
    if label[i]==0:
        plt.scatter(griddata[i,0], griddata[i,1],s=30, c="yellow", edgecolor="yellow")
    else:
        plt.scatter(griddata[i,0], griddata[i,1],s=30, c="green", edgecolor="green")

for i in range(0,len(data)):
        if data[i,2]==0:
            plt.scatter(data[i,0], data[i,1], s=50, c="yellow")
        else:
            plt.scatter(data[i,0], data[i,1], s=50, c="green")
plt.scatter(Cbound[:,0], Cbound[:,1], c='black')
plt.scatter(Cbound2[:,0], Cbound2[:,1], c='black')
fig1 = plt.gcf()
plt.show()
fig1.savefig('MAP.png', dpi=100)
