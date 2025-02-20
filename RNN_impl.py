'''
Package: RNN - Recurrent Neural Network
Author: Rahul Bhadani
What: Recurent Neural Network Implementation
'''

import numpy as np
import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from datetime import datetime
from RNN import RNN
from RNN import train_with_sgd
import rnn_utils
from RNN import generate_sentence

nltk.download('punkt')
vocabulary_size = 3000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
 
# Read the data and append SENTENCE_START and SENTENCE_END tokens
# CSV File can be downloaed from 
    # https://github.com/dennybritz/rnn-tutorial-rnnlm/blob/master/data/reddit-comments-2015-08.csv
print("Reading CSV file...")
with open('/home/ivory/VersionControl/RNN/reddit.csv', 'r', encoding="utf-8") as f:
    reader = csv.reader(f, skipinitialspace=True)
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print("Parsed %d sentences." % (len(sentences)))
# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print ("Found %d unique words tokens." % len(word_freq.items()))
# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
print ("Using vocabulary size %d." % vocabulary_size)
print ("The least frequent word in our vocabulary is '%s' and appeared %d times." 
% (vocab[-1][0], vocab[-1][1]))
# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
print( "\nExample sentence: '%s'" % sentences[0])
print ("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])

# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

np.random.seed(10)
model = RNN(vocabulary_size)
o, s = model.forward_propagation(X_train[10])

print(o.shape)
print(o)

predictions = model.predict(X_train[10])
print(predictions.shape)
print(predictions)

print("-------------------------------------------")
'''
Cross Entropy loss is L(y, o) = -\cfrac{1}{N}\sum_{n\in N} y_n \log o_n
'''
E_loss = np.log(vocabulary_size)
print('Expected loss for random predictions is {}'.format(E_loss))

Actual_loos = model.calc_loss(X_train[:1000], y_train[:1000])
print('Actual loss for random predictions is {}'.format(Actual_loos))

losses = train_with_sgd(model, X_train[:100], y_train[:100], nepoch=10, evaluate_loss_after=1)

 

### Lets generate the text now
num_sentences = 10
senten_min_length = 7

for i in range(num_sentences):
    sent = []

    while len(sent) < senten_min_length:
        sent = generate_sentence(model, word_to_index, index_to_word,
             sentence_start_token, sentence_end_token, unknown_token)
    print (' '.join(sent))
