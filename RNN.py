'''
Package: RNN - Recurrent Neural Network
Author: Rahul Bhadani
What: Preparing training dataset
'''
import numpy as np
import rnn_utils
class RNN:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        #Assign instance variable
        self.word_dim = word_dim #Number of words
        self.hidden_dim = hidden_dim #Number of hidden layers
        self.bptt_truncate = bptt_truncate

        #randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))

def forward_propagation(self, x):

    #The total number of time steps
    T = len(x)

    #During forward prop, we save all the hidden states in s as we need them later. We add one additional element for the initial hidden layer, which we set to 0.
    s = np.zeros((T+1, self.hidden_dim))
    s[-1] = np.zeros(self.hidden_dim) #Set the last vector to zeros.

    #The output at each time step. We save them for later.
    o = np.zeros((T, self.word_dim))

    #For each time step:
    for t in np.arange(T):
        #Note that we are indexing U by x[t]. This is the same as multplying U with a one-hot vector
        '''
        s_t = \tanh(Ux_t + Ws_{t-1})
        '''
        s[t] = np.tanh(self.U[:, x[t]]+ self.W.dot(s[t-1])) #since xt is a one-hot vector, we just need return a column of U for which index of x is 1
        '''
        o_t = \textsf{softmax}(Vs_t)
        '''
        o[t] = rnn_utils.softmax(self.V.dot(s[t]))
    return [o, s]

RNN.forward_propagation = forward_propagation

def predict(self, x):
    #Perform forward propagation and return index of the highest score
    o, s = self.forward_propagation(x)
    return np.argmax(o, axis=1) #Argmax returns the indices of the maximum values along an axis.

RNN.predict = predict
