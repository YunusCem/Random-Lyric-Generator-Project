from google.colab import drive 
import random
import numpy as np
from editdistance import eval 
import math

def fixChars(text):
    for char in [".",",","!","?",":",";","(",")","-","'",'"','[',']']:
        text = text.replace(char,"")
    for char in ['0','1','2','3','4','5','6','7','8','9']: 
        text = text.replace(char,"")
    text = text.replace("  ", " ")
    return text
    
def levenshteinSimilarity(a,b):
    sim = (max(len(a),len(b))-eval(a,b))/float(max(len(a),len(b)))
    return sim

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

def markov_prev2(curr1, curr2, prob_dict, prob_dict1, wordprob):
    curr = curr1+' '+curr2
    if curr not in prob_dict:
        return markov_prev1(curr1,prob_dict1,wordprob)
    else:
        pred_probs = prob_dict[curr]
        rand_prob = random.random()
        curr_prob = 0.0
        for prev in pred_probs:
            curr_prob += pred_probs[prev]
            if rand_prob <= curr_prob:
                return prev

def readLyrics(raw_text, bw_freqdict1={}, bw_freqdict2={}, wordfreq={}):
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

        for curr, succ1, succ2 in list(set(zip(words[:-2], words[1:-1], words[2:]))):
            succs = succ1+' '+succ2
            if succs not in bw_freqdict2:
                bw_freqdict2[succs] = {curr: 1}
            else:
               if curr not in bw_freqdict2[succs]:
                    bw_freqdict2[succs][curr] = 1
               else:
                    bw_freqdict2[succs][curr] += 1
    
    bw_probdict1 = {}
    for succ, succ_dict in bw_freqdict1.items():
        bw_probdict1[succ] = {}
        succ_total = sum(succ_dict.values())
        for curr in succ_dict:
            bw_probdict1[succ][curr] = float(succ_dict[curr]) / succ_total

    bw_probdict2 = {}
    for succ, succ_dict in bw_freqdict2.items():
        bw_probdict2[succ] = {}
        succ_total = sum(succ_dict.values())
        for curr in succ_dict:
            bw_probdict2[succ][curr] = float(succ_dict[curr]) / succ_total

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
    
    return bw_probdict1, bw_probdict2, wordprob

def markovlyrics1(wordprob, bw_probdict, nlines, sdlines, nwords, sdwords):
    m = list(wordprob.keys())
    prob = list(wordprob.values())
    lyr = np.random.choice(m,p=prob)
    lyrics = [lyr]
    l = 1 
    w = 1 
    linesize = np.random.normal(loc=nlines,scale=sdlines)
    while l <= linesize:
        if l == 1:
            threshold = np.random.normal(loc=nwords,scale=sdwords) 
            if w < threshold:
                lyrics.insert(-w,markov_prev1(lyrics[-w],bw_probdict,wordprob))
                w += 1
            else:
                lyrics.append("\n")
                w = 0
                l += 1
        elif (l % 2) == 0:
            if lyrics[-1] == "\n":
                prev = np.random.choice(m,p=prob)
                imp1 = 0
                imp = 0
                if len(lyrics[-2]) > 1 and len(prev) > 1:    
                    orrhyme = lyrics[-2][-2] + lyrics[-2][-1]
                    newrhyme = prev[-2] + prev[-1]
                elif len(lyrics[-2]) > 1 and len(prev) == 1:    
                    orrhyme = lyrics[-2][-2] + lyrics[-2][-1]
                    addword = markov_prev1(prev,bw_probdict,wordprob)
                    newrhyme = addword[-1] + prev[-1] 
                    imp1 = 1    
                elif len(lyrics[-2]) == 1 and len(prev) > 1:    
                    orrhyme = lyrics[-3][-1] + lyrics[-2][-1]
                    newrhyme = prev[-2] + prev[-1] 
                elif len(lyrics[-2]) == 1 and len(prev) == 1:
                    orrhyme = lyrics[-3][-1] + lyrics[-2][-1]
                    addword = markov_prev1(prev,bw_probdict,wordprob)
                    newrhyme = addword[-1] + prev[-1] 
                    imp1 = 1
                sim = levenshteinSimilarity(orrhyme,newrhyme)
                reps = 0
                while sim < 1-reps/2500 and reps<2500: 
                    imp = 0
                    prev = np.random.choice(m,p=prob)
                    if len(lyrics[-2]) > 1 and len(prev) > 1:    
                        orrhyme = lyrics[-2][-2] + lyrics[-2][-1]
                        newrhyme = prev[-2] + prev[-1]
                    elif len(lyrics[-2]) > 1 and len(prev) == 1:    
                        orrhyme = lyrics[-2][-2] + lyrics[-2][-1]
                        addword = markov_prev1(prev,bw_probdict,wordprob)
                        newrhyme = addword[-1] + prev[-1]
                        imp = 1
                    elif len(lyrics[-2]) == 1 and len(prev) > 1:
                        orrhyme = lyrics[-3][-1] + lyrics[-2][-1]
                        newrhyme = prev[-2] + prev[-1]
                    elif len(lyrics[-2]) == 1 and len(prev) == 1:
                        orrhyme = lyrics[-3][-1] + lyrics[-2][-1]
                        addword = markov_prev1(prev,bw_probdict,wordprob)
                        newrhyme = addword[-1] + prev[-1] 
                        imp = 1
                    sim = levenshteinSimilarity(orrhyme,newrhyme)
                    reps += 1
                if imp == 0:
                    lyrics.append(prev)
                elif imp == 1:
                    lyrics.append(addword)
                    lyrics.append(prev)
                    w += 1
                elif imp1 == 1:
                    lyrics.append(addword)
                    lyrics.append(prev)
                    w += 1
                imp = 0
                imp1 = 0
                w += 1                
            else:
                threshold = np.random.normal(loc=nwords,scale=sdwords)  
                if w < threshold:
                    lyrics.insert(-w,markov_prev1(lyrics[-w],bw_probdict,wordprob))
                    w += 1
                else:
                    lyrics.append("\n")
                    w = 0
                    l += 1
        elif (l % 2) == 1:
            threshold = np.random.normal(loc=nwords,scale=sdwords)
            if w == 0:
                lyrics.append(np.random.choice(m,p=prob))
                w += 1
            elif w < threshold:
                lyrics.insert(-w,markov_prev1(lyrics[-w],bw_probdict,wordprob))
                w += 1
            else:
                w = 0
                l += 1   
                lyrics.append("\n")                
    lyrics = " ".join(lyrics)
    return lyrics.replace("\n ", "\n")

def markovlyrics2(wordprob, bw_probdict, bw_probdict1, nlines, sdlines, nwords, sdwords):
    m = list(wordprob.keys())
    prob = list(wordprob.values())
    lyr1 = np.random.choice(m,p=prob)
    lyr2 = markov_prev1(lyr1,bw_probdict1,wordprob)
    lyr3 = 'thing'
    lyrics = [lyr1, lyr2, lyr3]
    l = 1 
    w = 2
    linesize = np.random.normal(loc=nlines,scale=sdlines)
    while l <= linesize:
        if l == 1:
            threshold = np.random.normal(loc=nwords,scale=sdwords) 
            if w < threshold:
                lyrics.insert(-w,markov_prev2(lyrics[-w],lyrics[-(w+1)],bw_probdict,bw_probdict1,wordprob))
                w += 1
            else:
                del lyrics[-1]
                lyrics.append("\n")
                w = 0
                l += 1 
        elif (l % 2) == 0:
            if lyrics[-1] == "\n":
                prev = np.random.choice(m,p=prob)
                addword = markov_prev1(prev,bw_probdict1,wordprob)
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
                reps = 0
                while sim < 1-reps/2500 and reps<2500: 
                    imp = 0
                    prev = np.random.choice(m,p=prob)
                    addword = markov_prev1(prev,bw_probdict1,wordprob)
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
                lyrics.append(addword)
                lyrics.append(prev)
                w += 2            
            else:
                threshold = np.random.normal(loc=nwords,scale=sdwords)  
                if w < threshold:
                    lyrics.insert(-w,markov_prev2(lyrics[-w],lyrics[-(w+1)],bw_probdict,bw_probdict1,wordprob))
                    w += 1
                else:
                    lyrics.append("\n")
                    w = 0
                    l += 1
        elif (l % 2) == 1:
            threshold = np.random.normal(loc=nwords,scale=sdwords)
            if w == 0:
                prev = np.random.choice(m,p=prob)
                addword = markov_prev1(prev,bw_probdict1,wordprob)
                lyrics.append(addword)
                lyrics.append(prev)
                w += 2
            elif w < threshold:
                lyrics.insert(-w,markov_prev2(lyrics[-w],lyrics[-(w+1)],bw_probdict,bw_probdict1,wordprob))
                w += 1
            else:
                w = 0
                l += 1   
                lyrics.append("\n")                
    lyrics = " ".join(lyrics)
    return lyrics.replace("\n ", "\n")

def Markovsong(wordprob, bw_probdict, bw_probdict1, nlines, sdlines, nwords, sdwords, repetition, sections, loops, loc1, loc2):
    store1 = ''
    chorus1 = ''
    for j in range(loops):
        r1 = [repetition] * (sections)
        r2 = [0] * (sections)
        for i in range(sections):
            r2[i] = np.random.uniform()
        for i in range(sections):
            if i == 1:
                chorus1 = ''
                p1 = markovlyrics1(wordprob, bw_probdict, nlines, sdlines, nwords, sdwords)+'\n'
                chorus1 = p1
                store1 = store1 + p1
            elif r1[i-1] > r2[i-1]:
                store1 = store1 + chorus1
            else:
                p1 = ''
                p1 = markovlyrics1(wordprob, bw_probdict, nlines, sdlines, nwords, sdwords)+'\n'
                store1 = store1 + p1
        chorus1 = ''
        p1 = ''
        store1 = store1 + '\n'
    #Save Lyrics in .txt file
    store1 = store1.split('/n')
    with open(loc1, 'w',encoding="utf-8") as filehandle:  
        for listitem in store1:
           filehandle.write('%s\n' % listitem)
    store2 = ''
    chorus2 = ''
    for j in range(loops):
        r1 = [repetition] * (sections)
        r2 = [0] * (sections)
        for i in range(sections):
            r2[i] = np.random.uniform()
        for i in range(sections):
            if i == 1:
                p2 = markovlyrics2(wordprob, bw_probdict, bw_probdict1, nlines, sdlines, nwords, sdwords)+'\n'
                chorus2 = p2
                store2 = store2 + p2
            elif r1[i-1] > r2[i-1]:
                store2 = store2 + chorus2
            else:
                p2 = ''
                p2 = markovlyrics2(wordprob, bw_probdict, bw_probdict1, nlines, sdlines, nwords, sdwords)+'\n'
                store2 = store2 + p2
        chorus2 = ''
        p2 = ''
        store2 = store2 + '\n'
    #Save Lyrics in .txt file
    store2 = store2.split('/n')
    with open(loc2, 'w',encoding="utf-8") as filehandle:  
        for listitem in store2:
           filehandle.write('%s\n' % listitem)
    return store1, store2

#Mounting Google drive
drive.mount('/content/drive')
#There is a aa bb cc type rhyming system programmed.
#Variables
numberoflines = [5.327999275,9.27547725]
sdoflines = [2.13126903,5.56688943]
sdofwords = [1.850334704,1.457065597]
numberofwords = [7.086869034,7.17317498]
numberofsections = [11,9]
repetitionscore = [0.519244002,0.263648001]
loops = [10,10]
loc1 = ['/content/drive/My Drive/Generated lyrics/Markov1s1.txt','/content/drive/My Drive/Generated lyrics/Markov1s2.txt']
loc2 = ['/content/drive/My Drive/Generated lyrics/Markov2s1.txt','/content/drive/My Drive/Generated lyrics/Markov2s2.txt']
raw_text = open('/content/drive/My Drive/lyricsText.txt', encoding = 'UTF-8').read()
lyric_bwprobdict1, lyric_bwprobdict2, probabilities = readLyrics(raw_text)
p11, p12 = Markovsong(probabilities, lyric_bwprobdict2, lyric_bwprobdict1, numberoflines[0], sdoflines[0], numberofwords[0], sdofwords[0], repetitionscore[0], numberofsections[0], loops[0], loc1[0], loc2[0])
p21, p22 = Markovsong(probabilities, lyric_bwprobdict2, lyric_bwprobdict1, numberoflines[1], sdoflines[1], numberofwords[1], sdofwords[1], repetitionscore[1], numberofsections[1], loops[1], loc1[1], loc2[1])