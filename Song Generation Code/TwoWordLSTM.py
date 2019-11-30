#Manually Coding LSTM with 2 Words

#Import the packages
from google.colab import drive 
import numpy as np
import pandas as pd
import sys
import keras
import tensorflow
from keras.models import Sequential
from keras.layers import Activation, Flatten, Dropout, Dense, Embedding, TimeDistributed, CuDNNLSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import RMSprop
from google.colab import drive 
import random
from editdistance import eval 
import math

def fixChars(text):
    for char in [".",",","!","?",":",";","(",")","-","'",'"']:
        text = text.replace(char,"")
    for char in ['0','1','2','3','4','5','6','7','8','9']: 
        text = text.replace(char,"")
    text = text.replace("  ", " ")
    return text
    
def levenshteinSimilarity(a,b):
    sim = (max(len(a),len(b))-eval(a,b))/float(max(len(a),len(b)))
    return sim

def readLyrics(raw_text):
    text = fixChars(raw_text)
    lines = text.replace("\n\n","\n").strip().lower().split("\n")
    word = []
    x1 = []
    y1 = []
    x21 = []
    x22 = []
    y2 = []
    for l in lines:
        words = l.strip().replace("  "," ").split()
        for i in list(set(words)):
            word.append(i)
        for curr, succ in list(set(zip(words[:-1], words[1:]))):
            x1.append(succ)
            y1.append(curr)
        for curr, succ1, succ2 in list(set(zip(words[:-2],words[1:-1], words[2:]))):
            x21.append(succ1) 
            x22.append(succ2)
            y2.append(curr)
      
    return x1, x21, x22, y1, y2, word

def markov_prev1(curr, prob_dict, wordprob):
    if curr not in prob_dict:
        m = list(wordprob.keys())
        prob = list(wordprob.values())
        return np.random.choice(m,p=prob)
    else:
        pred_probs = prob_dict[curr]
        rand_prob = random.random()
        curr_prob = 0.0
        for prev in pred_probs:
            curr_prob += pred_probs[prev]
            if rand_prob <= curr_prob:
                return prev

def readLyric(raw_text, bw_freqdict1={}, bw_freqdict2={}, wordfreq={}):
    text = fixChars(raw_text)
    lines = text.replace("\n\n","\n").strip().lower().split("\n")
    for l in lines:
        words = l.strip().replace("  "," ").split()
        for curr, succ in list(set(zip(words[:-1], words[1:]))):
            if succ not in bw_freqdict1:
                bw_freqdict1[succ] = {curr: 1}
            else:
                if curr not in bw_freqdict1[succ]:
                    bw_freqdict1[succ][curr] = 1
                else:
                    bw_freqdict1[succ][curr] += 1
    
    bw_probdict1 = {}
    for succ, succ_dict in bw_freqdict1.items():
        bw_probdict1[succ] = {}
        succ_total = sum(succ_dict.values())
        for curr in succ_dict:
            bw_probdict1[succ][curr] = float(succ_dict[curr]) / succ_total

    text = fixChars(raw_text)
    lines = text.replace("\n\n","\n").strip().lower().split("\n")
    for l in lines:
        words = l.strip().replace("  "," ").split()
        for curr in list(words):
            if curr not in wordfreq:
                wordfreq[curr] = 1
            else:
                wordfreq[curr] += 1
    wordprob = {}
    curr_total = sum(wordfreq.values())
    for curr in wordfreq:
            wordprob[curr] = float(wordfreq[curr]) / curr_total
    
    return bw_probdict1, wordprob

def getvalues(t, model):
    t = np.asarray(t)
    t = t/n_words
    t = t.reshape(1,2,1)
    t = model.predict(t)
    t = t.reshape(3406)
    t = t.tolist()
    t = t.index(max(t))
    return int_words[t]
  
#Starts with the txt file
#This will prompt authorization.
drive.mount('/content/drive')

#Load the dataset:
raw_text = open('/content/drive/My Drive/lyricsText.txt', encoding = 'UTF-8').read()

#There is a aa bb cc type rhyming system programmed unless there is a repetition.
#Variables
numberoflines = 10
sdoflines = 2
sdofwords = 1
numberofwords = 7
numberofsections = 2
#I will treat the repetition score as a scale from 0 to 1 where 1 is the same line repeated all the time for the whole song and 0 is all lines being seperate
repetitionscore = 0.0
loops = 5
raw_text = open('/content/drive/My Drive/lyricsText.txt', encoding = 'UTF-8').read()
x1, x21, x22, y1, y2, words = readLyrics(raw_text)
bw_probdict, probabilities = readLyric(raw_text)

#Mapping chars to ints:
words = list(set(words))
int_words = dict((i, c) for i, c in enumerate(words))
words_int = dict((i, c) for c, i in enumerate(words))

n_words = len(words)

#Process the dataset:
data_X1 = []
data_y1 = []
data_X21 = []
data_X22 = []
data_y2 = []

for i in range(len(y1)):
    #Store samples in data_X
    data_X1.append(words_int[x1[i]])
    #Store targets in data_y
    data_y1.append(words_int[y1[i]])
n_patterns1 = len(data_X1)

for i in range(len(y2)):
    #Store samples in data_X
    data_X21.append(words_int[x21[i]])
    #Store samples in data_X
    data_X22.append(words_int[x22[i]])
    #Store targets in data_y
    data_y2.append(words_int[y2[i]])
n_patterns2 = len(data_X21)

#Reshape X to be suitable to go into LSTM RNN :
X1 = np.reshape(data_X1 , (n_patterns1, 1))
X21 = np.reshape(data_X21 , (n_patterns2, 1))
X22 = np.reshape(data_X22 , (n_patterns2, 1))

#Normalizing input data : 
X1 = np.asarray(X1)
X21 = np.asarray(X21)
X22 = np.asarray(X22)
X1 = X1.reshape(n_patterns1,1,1)
X1 = X1/n_words

#Combining X2
X2 = np.concatenate((X21,X22),axis=1)
X2 = X2.reshape(n_patterns2,2,1)
X2 = X2/n_words

#One hot encode the output targets :
Y1 = np_utils.to_categorical(data_y1)
Y2 = np_utils.to_categorical(data_y2)
Y1 = np.asarray(Y1)
Y2 = np.asarray(Y2)

#Number of layers
layer_num = 6
#Number of nodes in each layer
layer_size = [256,256,256,256,256,256] 

#Specifying a sequential model for keras
model1 = Sequential()
#Adding an input layer
model1.add(CuDNNLSTM(layer_size[0], input_shape =(2, 1), return_sequences = True))
#Adding the four layers
for i in range(1,layer_num) :
    model1.add(CuDNNLSTM(layer_size[i], return_sequences=True))
#Adding a final layer
model1.add(Flatten())
model1.add(Dense(Y1.shape[1]))
model1.add(Activation('softmax'))
model1.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

#Configure the checkpoint (weights)
checkpoint_name = 'Weights-improvement-{epoch:03d}-{loss:.5f}.hdf5'
checkpoint = ModelCheckpoint(checkpoint_name, monitor='loss', verbose = 1, save_best_only = True, mode ='min')
callbacks_list = [checkpoint]

# Fit the model :
model_params = {'epochs':10,
                'batch_size':256,
                'callbacks':callbacks_list,
                'verbose':1,
                'validation_split':0.1,
                'validation_data':None,
                'shuffle': True,
                'initial_epoch':0,
                'steps_per_epoch':None,
                'validation_steps':None}
model1.fit(X2, 
          Y2,
          epochs = model_params['epochs'],
           batch_size = model_params['batch_size'],
           callbacks= model_params['callbacks'],
           verbose = model_params['verbose'],
           validation_split = model_params['validation_split'],
           validation_data = model_params['validation_data'],
           shuffle = model_params['shuffle'],
           initial_epoch = model_params['initial_epoch'],
           steps_per_epoch = model_params['steps_per_epoch'],
           validation_steps = model_params['validation_steps'])

#Setting up the model
#Set a random seed of 1 word:
#Generate Charachters :
linesize = np.random.normal(loc=numberoflines,scale=sdoflines) 
m = list(probabilities.keys())
prob = list(probabilities.values())
lyric1 = np.random.choice(m,p=prob)
lyric2 = markov_prev1(lyric,bw_probdict,probabilities)
lyric3 = 'thing'
lyrics = [lyric1,lyric2,lyric3]
r1 = [repetitionscore] * int(linesize)    
r2 = [0] * int(linesize)
for i in range(int(linesize)):
    r2[i] = np.random.uniform()
w = 1
l = 1
while l <= linesize:
    if l == 1:
        threshold = np.random.normal(loc=numberofwords,scale=sdofwords) 
        if w < threshold:
            t1 = words_int[lyrics[-w]]
            t2 = words_int[lyrics[-w+1]]
            t = [t1,t2]
            lyrics.insert(-w,getvalues(t, model1))
            w += 1
        else:
            del lyrics[-1]
            lyrics.append('\n')
            w = 0
            l += 1
    elif r1[l-1]>r2[l-1]:
        lyrics.extend(lyrics[-(math.floor(threshold)+2):])  
        l += 1
    elif (l % 2) == 0:
        if lyrics[-1] == "\n":
            prev = np.random.choice(m,p=prob)
            addword = markov_prev1(prev,bw_probdict,probabilities)
            imp1 = 0
            imp = 0
            if len(lyrics[-2]) > 1 and len(prev) > 1:    
                orrhyme = lyrics[-2][-2] + lyrics[-2][-1]
                newrhyme = prev[-2] + prev[-1]
            elif len(lyrics[-2]) > 1 and len(prev) == 1:    
                orrhyme = lyrics[-2][-2] + lyrics[-2][-1]
                newrhyme = addword[-1] + prev[-1]  
            elif len(lyrics[-2]) == 1 and len(prev) > 1:    
                orrhyme = lyrics[-3][-1] + lyrics[-2][-1]
                newrhyme = prev[-2] + prev[-1] 
            elif len(lyrics[-2]) == 1 and len(prev) == 1:
                orrhyme = lyrics[-3][-1] + lyrics[-2][-1]
                addword = markov_prev1(prev,bw_probdict,probabilities)
                newrhyme = addword[-1] + prev[-1] 
            sim = levenshteinSimilarity(orrhyme,newrhyme)
            reps = 0
            while sim < 1-reps/2500 and reps<2500: 
                imp = 0
                prev = np.random.choice(m,p=prob)
                addword = markov_prev1(prev,bw_probdict,probabilities)
                if len(lyrics[-2]) > 1 and len(prev) > 1:    
                    orrhyme = lyrics[-2][-2] + lyrics[-2][-1]
                    newrhyme = prev[-2] + prev[-1]
                elif len(lyrics[-2]) > 1 and len(prev) == 1:    
                    orrhyme = lyrics[-2][-2] + lyrics[-2][-1]
                    newrhyme = addword[-1] + prev[-1]
                elif len(lyrics[-2]) == 1 and len(prev) > 1:
                    orrhyme = lyrics[-3][-1] + lyrics[-2][-1]
                    newrhyme = prev[-2] + prev[-1]
                elif len(lyrics[-2]) == 1 and len(prev) == 1:
                    orrhyme = lyrics[-3][-1] + lyrics[-2][-1]
                    newrhyme = addword[-1] + prev[-1] 
                sim = levenshteinSimilarity(orrhyme,newrhyme)
                reps += 1
            if imp == 0:
                lyrics.append(addword)
                lyrics.append(prev)
            imp = 0
            imp1 = 0
            w += 1                
        else:
            threshold = np.random.normal(loc=numberofwords,scale=sdofwords)  
            if w < threshold:
                t1 = words_int[lyrics[-w]]
                t2 = words_int[lyrics[-w+1]]
                t = [t1,t2]
                lyrics.insert(-w,getvalues(t, model1))
                w += 1
            else:
                lyrics.append("\n")
                w = 0
                l += 1
    elif (l % 2) == 1:
        threshold = np.random.normal(loc=numberofwords,scale=sdofwords)
        if w == 0:
            lyrics.append(np.random.choice(m,p=prob))
            w += 1
        elif w < threshold:
            t1 = words_int[lyrics[-w]]
            t2 = words_int[lyrics[-w+1]]
            t = [t1,t2]
            lyrics.insert(-w,getvalues(t, model1))
            w += 1
        else:
            w = 0
            l += 1   
            lyrics.append("\n")                
lyrics = " ".join(lyrics)
lyrics = lyrics.replace("\n ", "\n")
print(lyrics)