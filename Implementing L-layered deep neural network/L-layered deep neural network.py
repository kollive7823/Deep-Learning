import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from sklearn import datasets
#from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    cache=z
    return s,cache

def relu(z):
    s = np.maximum(0,z)
    cache=z
    return s,cache

        
	
####### Helper functions ###########

def relu_derivative(dA, Z):
    """
    Implement the backward propagation for a single RELU unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_derivative(dA, Z):
    """
    Implement the backward propagation for a single SIGMOID unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    return dZ
    



#######################################################
    

#### Function to Initialize all the parameters
def initialize_parameters(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
		## storing the paramaters as a single python dictionary where the layer 1 parameters are "W1" and "b1"
        ### START CODE HERE ###
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01 ## random initialization ***multiply each parameter after random initialization with 0.01 to keep them small
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))## Initialize with zeros
        ### END CODE HERE ###
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.
    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    Z = W.dot(A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache 
	


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid","tanh" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing stored for computing the backward pass efficiently
    """
    #print(A_prev.shape)
    #print(W.shape)
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A"
        ### START CODE HERE ### 
        Z, linear_cache = linear_forward(A_prev, W, b)
        A,activation_cache = sigmoid(Z) ## compute sigmoid(W,b)
        ### END CODE HERE ###
        """    
    elif activation == "tanh":
        # Inputs: "A_prev, W, b". Outputs: "A".
        ### START CODE HERE ### 
        A = tanh(z) ## compute tanh(W,b) -- you can use a numpy function
        ### END CODE HERE ###
        """
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A,activation_cache= relu(Z)

    
	
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    
    cache = (linear_cache, activation_cache) ## Caching the variables to reuse them during back propagation

    return A, cache


	
	

### This function implements the entire forward propagation using the function above
def L_model_forward(X, parameters):
    """
    Implement forward propagation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- initialized parameters
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """
    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    # Implement (L-1) layers using RELU activation. Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        ### START CODE HERE ### 
        A, cache = linear_activation_forward(A_prev, 
                                             parameters['W' + str(l)], 
                                             parameters['b' + str(l)], 
                                             activation='relu') ## just call the right function with correct parameters (HINT: this function is already here)
		### also add 'cache' to the 'caches' list
        caches.append(cache)
        ### END CODE HERE ###
    
    # Implement the last layer computation using SIGMOID activation function. Add "cache" to the "caches" list.
    ### START CODE HERE ### 
    AL, cache = linear_activation_forward(A, 
                                          parameters['W' + str(L)], 
                                          parameters['b' + str(L)], 
                                          activation='sigmoid')
    caches.append(cache)
    
    ### END CODE HERE ###
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches
	
	
def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- (true) label vector [contains 0 and 1 for a binary classifier]

    Returns:
    cost -- using the cost function (also called the cross entropy cost)
    """
    
    m = Y.shape[1]

    
    ### START CODE HERE ### 
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))## Compute loss from AL (predicted label) and Y (true label).
    ### END CODE HERE ###
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost
	


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    return dA_prev, dW, db





	
	
def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values stored for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_derivative(dA, activation_cache)
        #dW = (np.dot(dZ, cache[0].T)) / m
        #db = (np.sum((dZ),axis=1,keepdims=True))/m
        #dA_prev = np.dot(cache[1].T, dZ) 
    elif activation =="sigmoid":
        dZ = sigmoid_derivative(dA, activation_cache)
        #dW = (np.dot(dZ, cache[0].T)) / m
        #db = np.sum((dZ),axis=1,keepdims=True) / m
        #dA_prev = np.dot(cache[1].T, dZ)
    """
        
       
    elif activation == "sigmoid":
        ### START CODE HERE ### 
        ## Call the right function given below (helper functions)
        dW = (np.dot(dZ, cache[0].T)) / m## Compute dW (HINT: the formula is given in the problem statement)
    	db = np.sum((dZ),axis=1,keepdims=True) / m  ### db computation is done here, look at the statement and understand it [refer to the formula given in the problem statement]
    	dA_prev = np.dot(cache[1].T, dZ)## Compute dA_prev (HINT: the formula is given in the problem statement)
   		### END CODE HERE ###
    """
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

	
### This function implements the entire backward propagation using the functions above
def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector 
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    # Initializing the backpropagation
	### START CODE HERE ###
    dAL = dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))## starting point of the backward prop. This is the derivative of the cost fucntion J(A,Y): A the output of forward prop. (predicted label) and Y is the true label. [refer to the formula given in the problem statement]
    ### END CODE HERE ###
	
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    ### START CODE HERE ### 
    current_cache = caches[L-1]## Reuses the variables from forward prop. [HINT: look at the 'caches' variable]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid") ## implement the backward step for the last layer [HINT:  the function is already there]
    ### END CODE HERE ###
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        ### START CODE HERE ### (approx. 5 lines)
        current_cache = caches[l]## Reuses the variables from forward prop. [HINT: look at the 'caches' variable]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")## implement the backward step at any layer [HINT:  the function is already there]
        grads["dA" + str(l + 1)] = dA_prev_temp## Assign the correct variable
        grads["dW" + str(l + 1)] = dW_temp## Assign the correct variable
        grads["db" + str(l + 1)] = db_temp## Assign the correct variable
        ### END CODE HERE ###

    return grads

#grads["dW" + str(3)].shape
#parameters["W" + str(1)].shape
	
	
	
def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    ### START CODE HERE ### 
    for l in range(1,L):
        parameters['W'+str(l+1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]## Update W (use the learning rate) [Hint: use the 'parameters' and 'grad' dictionaries]
        parameters['b'+str(l+1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]## Update b (use the learning rate) [Hint: use the 'parameters' and 'grad' dictionaries]
    ### END CODE HERE ###
    return parameters
	


if __name__ == '__main__':
	## suppose you have the input features in X (X.shape=(n,m)) and true label vector Y (Y.shape=(1,m))
	## you can use the same dataset from Assignment 1 (copy lines from 17-52 from the file Assignment1.py
	## set the structure of your network. You can vary the strudture and test. E.g.  [3,3,1] i.e. 1st and 2nd hidden layer has 3 units and the final layer has 1 unit
	## the activation used in the first two layers is RELU and the last one has SIGMOID
	### Here implement the entire training process of forward and backward prop.
	### all the functions you need are implemented above. You need to put them in the right sequence.
	### the other inputs you need are the hyperparameters -- learning rate, no. of iterations.
	
    ## Guidance
	
	## Inputs - (1) Data (X and Y) (2) Hyperparameters -- learning rate, number of iterations, network structure
	
	##Loading the data
    digits = datasets.load_digits()
    #digits[0]
    images_and_labels = list(zip(digits.images, digits.target))
    len(images_and_labels)
    train_set_x_orig = [] 
    train_set_y = []
    test_set_x_orig = [] 
    test_set_y = []
    classes = ['even','odd']
    
    
    data_size = len(images_and_labels)
    #Setting the testset size
    test_set_size = 200
    
    
    ##Splitting the data into training and test sets and assigning the labels: Even (0), Odd (1)
    
    for i in range(len(images_and_labels))[:-test_set_size]:
        train_set_x_orig.append(images_and_labels[i][0])
        if images_and_labels[i][1] %2 == 0: ##if even put 0 as the label
            train_set_y.append(0)
        else: ##if odd put 1 as the label
            train_set_y.append(1)
            
    for i in range(len(images_and_labels))[-test_set_size:]:
        test_set_x_orig.append(images_and_labels[i][0])
        if images_and_labels[i][1] %2 == 0: ##if even put 0 as the label
            test_set_y.append(0)
        else: ##if odd put 1 as the label
            test_set_y.append(1)
    
    train_set_x_orig = np.array(train_set_x_orig)
    train_set_y = np.array(train_set_y).reshape(1,data_size-test_set_size)
    test_set_x_orig = np.array(test_set_x_orig)
    test_set_y = np.array(test_set_y).reshape(1,test_set_size)
    
    print (train_set_x_orig.shape, train_set_y.shape, test_set_x_orig.shape, test_set_y.shape)
    
    ## Display an example from the data
    index = 1
    #plt.imshow(train_set_x_orig[index])
    print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])] +  "' number.")
    
    m_train = train_set_y.shape[1]
    m_test = test_set_y.shape[1]
    num_px = train_set_x_orig.shape[2]
    
    ##Description of the data
    print ("Number of training examples: m_train = " + str(m_train))
    print ("Number of testing examples: m_test = " + str(m_test))
    print ("Height/Width of each image: num_px = " + str(num_px))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ")")
    print ("train_set_x shape: " + str(train_set_x_orig.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x shape: " + str(test_set_x_orig.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))
    
    # Reshape the training and test examples from 2D matrix to a 1D vector. The input vector of a neural network is always 1D
    train_set_x_flatten = train_set_x_orig.reshape(m_train,(num_px*num_px)).T
    test_set_x_flatten = test_set_x_orig.reshape(m_test,(num_px*num_px)).T
    
    # Print the description after reshaping
    print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))
    
    # Standardize the pixel value
    train_set_x = train_set_x_flatten/255.
    test_set_x = test_set_x_flatten/255.
	
	## Step 1. Initialize the parameters
    layer_dims = [num_px*num_px,3,4,1]
    learning_rate = 0.001
    parameters = initialize_parameters(layer_dims)
	
	## Step 2. Train for number of iterations
    num_iterations = 10
    for i in range(num_iterations):
        y_cap,caches= L_model_forward(train_set_x,parameters)
        cost = compute_cost(y_cap,train_set_y)
        print(cost)
        grads = L_model_backward(y_cap,train_set_y,caches)
        update_parameters(parameters,grads,learning_rate)

	## Step 2a. Forward propagation
        
    
	
	## Step 2b. Compute cost [this is not necessary as only the derivative of the cost is required but you can still compute cost to see if its dropping after each iterations]
	
	## Step 2c. Backward propagation
	
	## Step 2d. Update weights
##Loading the data
