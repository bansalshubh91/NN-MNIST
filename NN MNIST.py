
# coding: utf-8

# In[1]:


import numpy as np
import h5py
import time
import copy
from random import randint
#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()


#Implementation of stochastic gradient descent algorithm
#number of inputs
num_inputs = 28*28
#number of hidden units in hidden layer
num_hidden_units = 70
#number of outputs
num_outputs = 10
model = {}
model['W1'] = np.random.randn(num_hidden_units,num_inputs) / np.sqrt(num_inputs)
model['b1'] = np.random.randn(num_hidden_units,1) 

model['C'] = np.random.randn(num_outputs,num_hidden_units) / np.sqrt(num_hidden_units)
model['b2'] = np.random.randn(num_outputs,1) 

model_grads = copy.deepcopy(model)



def softmax_function(z):
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ
def forward(x,y, model):
    Z = np.dot(model['W1'], x).reshape((num_hidden_units,1)) + model['b1']
    activ_deriv = np.zeros(shape=Z.shape)
    for i in range(len(Z)): 
        if (Z[i]>=0) :
            activ_deriv[i] = 1
        else :
            activ_deriv[i] = 0
    H = np.zeros(shape=Z.shape)
    for i in range(len(Z)):
        H[i] = max(Z[i],0)
    U = np.dot(model['C'], H).reshape((num_outputs,1)) + model['b2']
    p = softmax_function(U)
    return p,H,activ_deriv 
def backward(x,y,p,H,activ_deriv, model, model_grads):
    dU = 1.0*p
    dU[y] = dU[y] - 1.0
    for i in range(num_outputs):
        model_grads['C'][i,:] = dU[i]*(H.reshape((num_hidden_units,)))
    model_grads['b2'] = dU    
    delta = np.dot(np.transpose(model['C']), dU)
    model_grads['W1'] = np.dot((delta*activ_deriv).reshape((num_hidden_units,1)), (np.transpose(x)).reshape((1,num_inputs)))
    model_grads['b1'] = (delta*activ_deriv).reshape((num_hidden_units,1))
    
    return model_grads


import time
time1 = time.time()
LR = .01
num_epochs = 30
for epochs in range(num_epochs):
    #Learning rate schedule
    if (epochs > 5):
        LR = 0.001
    if (epochs > 10):
        LR = 0.0001
    if (epochs > 15):
        LR = 0.00001
    total_correct = 0
    for n in range( len(x_train)):
        n_random = randint(0,len(x_train)-1 )
        y = y_train[n_random]
        x = x_train[n_random][:]
        p,H,activ_deriv = forward(x, y, model)
        prediction = np.argmax(p)
        if (prediction == y):
            total_correct += 1
        model_grads = backward(x,y,p,H,activ_deriv, model, model_grads)
        model['W1'] = model['W1'] - LR*model_grads['W1']
        model['C'] = model['C'] - LR*model_grads['C']
        model['b1'] = model['b1'] - LR*model_grads['b1']
        model['b2'] = model['b2'] - LR*model_grads['b2']

    print(total_correct/np.float(len(x_train) ) )
time2 = time.time()
print(time2-time1)
######################################################
#test data
total_correct = 0
for n in range( len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    p,H,activ_deriv = forward(x, y, model)
    prediction = np.argmax(p)
    if (prediction == y):
        total_correct += 1
print(total_correct/np.float(len(x_test) ) )


