#LSTM Using the textgenrnn Package

#Import Packages
from textgenrnn import textgenrnn
from google.colab import drive
import os

def trainLSTM(path,epochs,genepochs,trainratio,layers,perceptrons,bidirectional,length,maxlength):
    textgen = textgenrnn()
    textgen.train_from_file(
        file_path=path,
        new_model=True,
        num_epochs=epochs,
        train_size=trainratio,
        rnn_bidirectional=True,
        max_length=length,
        word_level=True)
    return textgen

def listToString(s):
    str1 = "" 
    for w in s:  
        str1 += w
    return str1  

def fixChars(text):
    for char in [".",",","!","?",":",";","(",")","-","'",'"','[',']']:
        text = text.replace(char,"")
    for char in ['0','1','2','3','4','5','6','7','8','9']: 
        text = text.replace(char,"")
    text = text.replace("  ", " ")
    return text

def readLyrics(raw_text):
    text = fixChars(raw_text)
    lines = text.replace("\n\n","\n").strip().lower().split("\n")
    for l in lines:
        words = l.strip().replace("  "," ").split()
    return lines, words


def genSection(textgen, nlines, sdlines, nwords, sdwords):
    l = 1 
    w = 0
    linesize = np.random.normal(loc=nlines,scale=sdlines)
    words = []
    while len(words) < 300:
        rlines = textgen.generate(temperature=1.0, return_as_list=True)
        rlines = listToString(rlines)
        words = words + rlines.strip().replace("  "," ").split()
    try:
        words.remove('\n')
    except:
        pass
    try:
        words.remove(' ')
    except:
        pass
    try:
        words.remove('')
    except:
        pass
    lyrics = []
    threshold = 0
    while l-1 <= linesize:
        if w < threshold:
            lyrics.append(words[w])
            w += 1
        else:
            lyrics.append("\n")
            l += 1
            threshold = np.random.normal(loc=nwords,scale=sdwords) + threshold
    threshold = 0
    lyrics = " ".join(lyrics)
    lyrics = lyrics.replace("\n\n", "\n")
    lyrics = lyrics.replace("\n\n", "\n")
    lyrics = lyrics.replace("\n",  "\n ")
    return lyrics

def LSTMsong(textgen, nlines, sdlines, nwords, sdwords, repetition, sections, loops, loc):
    store = ''
    chorus = ''
    for j in range(loops):
        r1 = [repetition] * (sections)
        r2 = [0] * (sections)
        for i in range(sections):
            r2[i] = np.random.uniform()
        for i in range(sections):
            if i == 1:
                chorus = ''
                p = genSection(textgen, nlines, sdlines, nwords, sdwords)+'\n'
                chorus = p
                store = store + p
            elif r1[i-1] > r2[i-1]:
                store = store + chorus
            else:
                p = ''
                p = genSection(textgen, nlines, sdlines, nwords, sdwords)+'\n'
                store = store + p
        chorus = ''
        p = ''
        store = store + '\n'
    #Save Lyrics in .txt file
    store = store.split('/n')
    with open(loc, 'w',encoding="utf-8") as filehandle:  
        for listitem in store:
           filehandle.write('%s\n' % listitem)
    return store

#This will prompt authorization.
drive.mount('/content/drive')

#Variables
numberoflines = [5.327999275,9.27547725]
sdoflines = [2.13126903,5.56688943]
sdofwords = [1.850334704,1.457065597]
numberofwords = [7.086869034,7.17317498]
numberofsections = [11,9]
repetitionscore = [0.519244002,0.263648001]
loops = [10,10]
loc = ['/content/drive/My Drive/Generated lyrics/LSTMRNNs1.txt','/content/drive/My Drive/Generated lyrics/LSTMRNNs2.txt']

#Fixing the words and saving the new file
path = '/content/drive/My Drive/lyricsfixed.txt'
raw_text = open('/content/drive/My Drive/lyricsText.txt', encoding = 'UTF-8').read()
lines, words = readLyrics(raw_text)
with open(path, 'w',encoding="utf-8") as filehandle:  
        for listitem in lines:
           filehandle.write('%s\n' % listitem)

#Training LSTM
textgen = trainLSTM('/content/drive/My Drive/lyricsfixed.txt',20,4,0.8,5,250,True,10,10000)

#Generating and saving the songs
p1 = LSTMsong(textgen, numberoflines[0], sdoflines[0], numberofwords[0], sdofwords[0], repetitionscore[0], numberofsections[0], loops[0], loc[0])
p2 = LSTMsong(textgen, numberoflines[1], sdoflines[1], numberofwords[1], sdofwords[1], repetitionscore[1], numberofsections[1], loops[1], loc[1])