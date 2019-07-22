'''
Package: RNN - Recurrent Neural Network
Author: Rahul Bhadani
What: Preparing training dataset
'''
import numpy as np
import rnn_utils
import operator
import datetime
import sys
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

def cross_entropy_loss(self, x, y):
    '''
    Function to calculate corss entropy loss
    L(y, o) = -\cfrac{1}{N}\sum_{n\in N} y_n \log o_n
    '''
    L = 0.0
    #For each sentence:
    for i in np.arange(len(y)):
        o, s = self.forward_propagation(x[i])
        
        #We only care about prediction of the correct words
        correct_word_predictions = o[np.arange(len(y[i])), y[i]]
        
        #Add the loss based on how difference is their from actual label
        L += -1*np.sum(np.log(correct_word_predictions))
    return L

def calc_loss(self, x, y, loss_type='cross_entropy'):
    '''
        #Calculate the loss. We will make room for different loss types
        But default loss is the cross entropy loss
    '''
    
    N = np.sum((len(y_i) for y_i in y))
    if loss_type == 'cross_entropy':
        #Divide the total loss by the number of training examples
        return self.cross_entropy_loss(x, y)/N
    else:
        return None

RNN.cross_entropy_loss = cross_entropy_loss
RNN.calc_loss = calc_loss

def bptt(self, x, y):
    T = len(y)

    #Perform forward propagation
    o, s = self.forward_propagation(x)
    #We accumulate the gradients in these variables
    dLdU = np.zeros(self.U.shape)
    dLdV = np.zeros(self.V.shape)
    dLdW = np.zeros(self.W.shape)

    delta_o = o
    delta_o[np.arange(len(y)), y] -= 1.

    # For each outout backwards:
    # [::-1] means reverse the list
    for t in np.arange(T)[::-1]:
        dLdV += np.outer(delta_o[t], s[t].T) #Outer calculates the product of two vectors

        #Initial delta calculation
        delta_t = self.V.T.dot(delta_o[t])*(1- (s[t]**2))
        
        #Backpropagation through time (for at most self.bptt_truncate steps)
        for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
            dLdW += np.outer(delta_t, s[bptt_step - 1])
            dLdU[:, x[bptt_step]] +=delta_t

            #Update the delta for the next step
            delta_t = self.W.T.dot(delta_t)*(1-s[bptt_step-1]**2)
        return [dLdU, dLdV, dLdW]

RNN.bptt = bptt

def gradient_check(self, x, y, h=0.001, error_threshold=0.01):

    #Calculate the gradients using backprop, we want to check if these are correct
    bptt_gradients = self.bptt(x, y)

    #List of all parameters we wantr to check
    model_parameters = ['U', 'V', 'W']

    #Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        #get the actual parameter value from the model:
        parameter = operator.attrgetter(pname)(self)
        print('Performing gradient check on {}'.format(pname))

        #Iterate over each element of the parameter matrix
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            print('Idex = {}'.format(ix))
            #save the original value so we can reset it later
            original_value = parameter[ix]

            #Estimate the gradient using 
            '''
            \pdv{L}{\theta} \approx \lim_{h\to 0} \cfrac{L(\theta + h) - L(\theta - h)}{2h}
            '''
            parameter[ix] = original_value + h
            gradplus = self.cross_entropy_loss([x], [y])
            parameter[ix] = original_value - h
            gradminus = self.cross_entropy_loss([x], [y])
            estimated_gradient = (gradplus - gradminus)/(2.0*h)

            #Resent parameter to the original value
            parameter[ix] = original_value

            #The gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]

            #Calculate the relative Error (|x-y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
            print('Num: {}'.format(np.abs(backprop_gradient - estimated_gradient)))
            print('Denom: {}'.format((np.abs(backprop_gradient) + np.abs(estimated_gradient))))
            print('Backprop gradient: {}'.format(backprop_gradient))
            print('Estimated gradient: {}'.format(estimated_gradient))

            if relative_error > error_threshold:
                print('Gradient Check ERROR: parameter={0} , ix={1}'.format(pname, ix))
                print('+h loss: {}'.format(gradplus))
                print('-h loss: {}'.format(gradminus))
                print('Backprop gradient: {}'.format(backprop_gradient))
                print('Estimated gradient: {}'.format(estimated_gradient))
                print('Relative Error: {}'.format(relative_error))
                return
            it.iternext()
        print('Gradient Check for parameter {0} passed'.format(pname))

RNN.gradient_check = gradient_check

                
def sgd_step(self, x, y, learning_rate):
    '''
    Perform one step of SGD
    '''
    #Calculate the gradients
    dLdU, dLdV, dLdW = self.bptt(x, y)
    #Change the parameters according to the gradient 
    self.U -= learning_rate*dLdU
    self.V -= learning_rate*dLdV
    self.W -= learning_rate*dLdW

RNN.sgd_step = sgd_step

def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    '''
    This is outer SGD loop.
    model: The RNN Model Instance
    X_train: The training dataset
    y_train: The training label
    learning_rate: Initial learning rate for SGD
    nepoch: Number of episodes to iterate through the complete dataset
    evaluate_loss_after: Evaluate the loss after this many epochs
    '''
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        #Evaluate the loss once in a while
        if (epoch % evaluate_loss_after == 0):
            loss = model.calc_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('{}: Loss after {} examples, episoe = {} is {}'.format(time, num_examples_seen, epoch, loss))
            #Adjust the learning rate if loss increases   
            if (len(losses) > 1):
                if(losses[-1][1] > losses[-2][1]):
                    learning_rate = learning_rate*0.5
                    print('Setting learning rate to {}'.format(learning_rate))
            sys.stdout.flush()
        #For each training example:
        for i in range(len(y_train)):
            #ONE SGD Step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen +=1
    return losses

def generate_sentence(model, word_to_index, index_to_word, sentence_start_token, sentence_end_token, unknown_token):
    #We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        [next_word_probs, s] = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]

        #We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            nex_probs = next_word_probs[-1]
            samples = np.random.multinomial(1, nex_probs)
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str