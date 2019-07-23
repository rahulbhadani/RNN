# -*- coding: utf-8 -*-
"""Crypt_LSTM

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1guzTMibpzWywlckt9xu2gazsb-jifVDG
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import io
# %matplotlib inline

# Decode the files as uploaded file is a dictionary of keys and values
btc = pd.read_csv('/home/ivory/VersionControl/RNN/btc.csv')

# select only the coloumn we need in the prediction process
data_to_use = btc['Close'].values
btc.head()

# select bitcoin closing price as the target variable to predict
data_to_use=btc['Close'].values
data_to_use

# data preprocessing using sklearn to scale our data and then fit into our model
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_to_use.reshape(-1,1))

# Now we plot the data to see how the bitcoin close price tended over the given time period
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
fig.set_size_inches(10.0, 6.0)
ax.grid(which='major', linestyle='-', linewidth='0.75', color='white')
ax.grid(which='minor', linestyle=':', linewidth='0.25', color='black')
plt.title('Bitcoin prices from December 2014 to May 2018')
ax.set_facecolor((1.0, 0.17, 0.2, 0.35))
plt.xlabel('Days')

plt.ylabel('Scaled price of Bitcoin')
plt.plot(scaled_data, label='Price')

ax.spines['bottom'].set_color('None')
ax.spines['top'].set_color('None') 
ax.spines['right'].set_color('None')
ax.spines['left'].set_color('None')
plt.legend()
plt.show()

# Features and label dataset
# We define a function to create the features and labels for  our dataset by windowing the data

def window_data(data, window_size):
    '''
    Input:
        data - this is the dataset we use
        window_size - how many datapoints we use to predict the next datapoint in the seqeunce. 
    Output:
        X - features split into widnows of data points
        y - labels - this is the next number in the sequence that we are trying to predict
    '''
    X = []
    y = []
    i = 0
    while (i + window_size) <= len(data) - 1:
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
        
        i += 1
    assert len(X) == len(y)
    return X, y

X, y  = window_data(scaled_data, 7)

X_train = np.array(X[:1018])
y_train = np.array(y[:1018])

X_test = np.array(X[1018:])
y_test = np.array(y[1018:])

print('X_train size: {}'.format(X_train.shape))

print('y_train size: {}'.format(y_train.shape))

print('X_test size: {}'.format(X_test.shape))

print('y_test size: {}'.format(y_test.shape))

#  Define the network
batch_size = 7 #Number of windows of data we are passing at once
window_size = 7 #Number of days we consider while predict the bitcoin price
hidden_layer = 256 # The number of hidden layer /no. of units in our LSTM cell.
clip_margin = 4 #This prevents exploding the gradient - use clipper to clip the gradient below this margin
learning_rate = 0.001 #Learning rate that aims to reduce the loss function
epochs = 200 #The number of episodes our model needs to build itself

#Placeholders for tensorflow
inputs = tf.placeholder(tf.float32, [batch_size, window_size, 1])
targets = tf.placeholder(tf.float32, [batch_size, 1])

# LSTM Weights
# Since LSTM weights are determined by operation gates that includes forget, input and out gates,
'''
Forget Gate
f & = \sigma(x_t U^f + s_{t-1}W^f)
This is a sigmoid layer that takes the outout at t-1 time and the current input at t time
and combine them into a single tensor. Then, it applies a linear transformation followed
by a sigmoid.

Output of the forget gate is between 0 to 1 s we have a sigmoid function. The output is
multiplied with the internal state and that is why the gate is called as the forget gate.
If f = 0, then the previous internal state is completely forgotten, if f = 1, then it will
be pass unaltered.
'''

'''
Input Gate:
i & = \sigma(x_t U^i + s_{t-1}W^i)
This takes the previous output with a new input and passes them through a sigmoid layer.
The gate returns a value between 0 and 1. The value of the input gate is multiplied with
the output of the cadidate layer.
'''

'''
Candidate hidden state:
g & = \tanh(x_tU^g  + s_{t-1}W^g)
Applies hyperbolic tangent to the mix of the input and previous output, returns the candidate
vector. The candidate vector is then added to the internal state using the update rule:
c_t & = c_{t-1} \odot + g\odot i

The previous state is multiplied by the forget gate and then added to the fraction of
the new candidate allowed by the output gate.
'''

'''
Output Gate:
o & = \sigma(x_tU^o + s_{t-1}W^o)
s_t & = \tanh(c_t)\odot o

This controls how much of the internal state is passed to the output and works in a similar
fashion as other gates.

'''
# tf.truncated normal generates a normal distribution with specified mean and standard
# deviation except that values whose magnitude is more than 2.0 standard deviation from
# the mean are dropped and re-picked. Default mean is 0.0

#Weights for the input gate
weights_input_gate = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
weights_input_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
bias_input = tf.Variable(tf.zeros([hidden_layer]))

#weights for the forgot gate
weights_forget_gate = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
weights_forget_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
bias_forget = tf.Variable(tf.zeros([hidden_layer]))

#weights for the output gate
weights_output_gate = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
weights_output_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
bias_output = tf.Variable(tf.zeros([hidden_layer]))

#weights for the memory cell
weights_memory_cell = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
weights_memory_cell_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
bias_memory_cell = tf.Variable(tf.zeros([hidden_layer]))

#Output layer weigts
weights_output = tf.Variable(tf.truncated_normal([hidden_layer, 1], stddev=0.05))
bias_output_layer = tf.Variable(tf.zeros([1]))

def LSTM_cell(input, output, state):
    input_gate = tf.sigmoid(tf.matmul(input, weights_input_gate) 
                            + tf.matmul(output, weights_input_hidden) 
                            + bias_input) #i & = \sigma(x_t U^i + s_{t-1}W^i)
    forget_gate = tf.sigmoid(tf.matmul(input, weights_forget_gate) 
                            + tf.matmul(output, weights_forget_hidden) 
                            + bias_forget) #f & = \sigma(x_t U^f + s_{t-1}W^f)
    output_gate = tf.sigmoid(tf.matmul(input, weights_output_gate) 
                            + tf.matmul(output, weights_output_hidden) 
                            + bias_output) #\sigma(x_tU^o + s_{t-1}W^o)
    memory_cell = tf.tanh(tf.matmul(input, weights_memory_cell) 
                            + tf.matmul(output, weights_memory_cell_hidden) 
                            + bias_memory_cell) #\tanh(x_tU^g  + s_{t-1}W^g)
    
    state = state*forget_gate + input_gate*memory_cell #c_{t-1} \odot + g\odot i
    
    output = output_gate*tf.tanh(state)
    
    return state, output

# Network loop 
# A loop for network is created which iterates through every window in the batch
# creating the `batch_states` as all zeros. The output is used for predicting
# the bitcoin price

outputs = []

#iterates through every window in the batch:
for i in range(batch_size):
    #for each batch, I am creating batch_state as all zeros and output for that
    # window which is all zeros at the beginning as well
    batch_state = np.zeros([1, hidden_layer], dtype=np.float32)
    batch_output = np.zeros([1, hidden_layer], dtype=np.float32)
    
    # for each point in the window we are feeding into LSTM to get next output
    for ii in range(window_size):
        batch_state, batch_output = LSTM_cell(tf.reshape(inputs[i][ii], (-1, 1)), 
                                             batch_state, batch_output)
        
    #Last output is considered and used to get a prediction
    outputs.append(tf.matmul(batch_output, weights_output) + bias_output_layer)
    
outputs

# Define Loss
# We use `mean_squared_error` function for the loss to minimize the errors.

losses = []
for i in range(len(outputs)):
    losses.append(tf.losses.mean_squared_error(
        tf.reshape(targets[i], (-1, 1)), outputs[i]))
    
# Computes the mean of elements across dimensions of a tensor.
loss = tf.reduce_mean(losses) 

# We define the optimizer with gradient clipping
gradients = tf.gradients(loss, tf.trainable_variables())

# Clips values of multiple tensors by the ratio of the sum of their norms.
clipped, _ = tf.clip_by_global_norm(gradients, clip_margin)

optimizer = tf.train.AdamOptimizer(learning_rate)
trained_optimizer = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))

# Train the network
# We now train the network with the number of epochs = 200 and then observe
# change in our loss through the time.
# The current loss decreases with the increase in the epochs as observed
# increasing our model accuracy in predicting the bitcoin prices
session = tf.Session()
session.run(tf.global_variables_initializer())

for i in range(epochs):
    traind_scores = []
    ii = 0
    epoch_loss = []
    while(ii+batch_size) <= len(X_train):
        X_batch = X_train[ii:ii+batch_size]
        y_batch = y_train[ii:ii+batch_size]
        
        o, c, _ = session.run([outputs, loss, trained_optimizer], 
                              feed_dict = {inputs:X_batch, targets:y_batch})
        
        epoch_loss.append(c)
        traind_scores.append(o)
        ii += batch_size
    if (i%30) == 0:
        print('Epoch {}/{}'.format(i, epochs), 
              'Current loss: {}'.format(np.mean(epoch_loss)))

sup = []
for i in range(len(traind_scores)):
    for j in range(len(traind_scores[i])):
        sup.append(traind_scores[i][j][0])
        

tests = []
i = 0
while i+batch_size <= len(X_test):
    o = session.run([outputs], feed_dict={inputs:X_test[i:i+batch_size]})
    i += batch_size
    tests.append(o)
    
tests_new = []
for i in range(len(tests)):
      for j in range(len(tests[i][0])):
        tests_new.append(tests[i][0][j])

test_results = []
for i in range(1264):
    if i >= 1019:
        test_results.append(tests_new[i-1019])
    else:
        test_results.append(None)

# Plotting the predictions
fig, ax = plt.subplots()
fig.set_size_inches(10.0, 6.0)
ax.grid(which='major', linestyle='-', linewidth='0.75', color='white')
ax.grid(which='minor', linestyle=':', linewidth='0.25', color='black')
plt.title('Bitcoin prices from December 2014 to May 2018 - Prediction Plot')
ax.set_facecolor((1.0, 0.17, 0.2, 0.35))
plt.xlabel('Days')

plt.ylabel('Scaled price of Bitcoin')
plt.plot(scaled_data, label='Price')
plt.plot(sup, label='Training data')
plt.plot(test_results, label='Testing data')
ax.spines['bottom'].set_color('None')
ax.spines['top'].set_color('None') 
ax.spines['right'].set_color('None')
ax.spines['left'].set_color('None')
plt.legend()
plt.show()
