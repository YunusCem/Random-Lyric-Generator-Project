import pandas as pd
import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from matplotlib.pyplot import figure
from matplotlib import colors
cmap = colors.LinearSegmentedColormap('red_blue_classes', {'red': [(0, 1, 1), (1, 0.3, 0.3)], 'green': [(0, 0.6, 0.6), (1, 0.6, 0.6)], 'blue': [(0, 0.3, 0.3), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)

def normalize(x):
    return x/((x.max()-x.min())*1.0)

sample_size = 100;
data_dim = 6;
songs = ['000.txt','001.txt','002.txt','003.txt','004.txt','005.txt','006.txt','007.txt','008.txt','009.txt','010.txt',
         '011.txt','012.txt','013.txt','014.txt','015.txt','016.txt','017.txt','018.txt','019.txt','020.txt',
         '021.txt','022.txt','023.txt','024.txt','025.txt','026.txt','027.txt','028.txt','029.txt','030.txt',
         '031.txt','032.txt','033.txt','034.txt','035.txt','036.txt','037.txt','038.txt','039.txt','040.txt',
         '041.txt','042.txt','043.txt','044.txt','045.txt','046.txt','047.txt','048.txt','049.txt',
         '051.txt','052.txt','053.txt','054.txt','055.txt','056.txt','057.txt','058.txt','059.txt','060.txt',
         '061.txt','062.txt','063.txt','064.txt','065.txt','066.txt','067.txt','068.txt','069.txt','070.txt',
         '071.txt','072.txt','073.txt','074.txt','075.txt','076.txt','077.txt','078.txt','079.txt','080.txt',
         '081.txt','082.txt','083.txt','084.txt','085.txt','086.txt','087.txt','088.txt','089.txt','090.txt',
         '091.txt','092.txt','093.txt','094.txt','095.txt','096.txt','097.txt','098.txt','099.txt','100.txt']
shape = (sample_size, data_dim)
data_vec = np.empty(shape)

index = 0
#Extract data from songs
for song in songs:
    with open(song) as fp:
        begin = True
        bracket = False
        LpS_arr = np.empty(0)     #array of lines/section
        WpLpS_arr = np.empty(0)     #array of average words_per_line/section
        first_arr = np.empty(0)     #array for strings
        temp_WpL = np.empty(0)
        repeats = 0     # (num Sect. repeated)/(Total num Sect. - 1)
        num_sections = 1;
        words_in_line = 0;
        lines_in_section = 0;
        line = fp.readline()
        
        while line:

            # New Section
            if line == "\n":
                begin = True
                if bracket == False:
                    LpS_arr = np.append(LpS_arr,lines_in_section)         #add num lines for prev section
                    WpLpS_arr = np.append(WpLpS_arr, np.mean(temp_WpL))   #add average line length for prev section
                    temp_WpL = np.empty(0)                                #clear temp array
                    num_sections += 1;
                    lines_in_section = 0;
                    #print("^^^AVERAGE WORDS/LINE = ", WpLpS_arr)
                    #print("WHATS THE MEAN: ", np.mean(WpLpS_arr))
                    #print("BEGIN SECTION #", num_sections)
                    #print("Lines/Section", LpS_arr)
            
            # Not a real line
            elif line[0] == '[':
                bracket = True
                #Skip, do nothing
                #print(" ")
            
            # Cache the first line
            elif begin:
                first_arr = np.append(first_arr,line)
                begin = False
                bracket = False
                lines_in_section += 1;
                words_in_line = len(line.split())
                temp_WpL = np.append(temp_WpL,words_in_line)
                #print(line.strip(), words_in_line)
            
            # Extract Content
            else:
                bracket = False
                lines_in_section += 1;
                words_in_line = len(line.split())
                temp_WpL = np.append(temp_WpL,words_in_line)
                #print(line.strip(), words_in_line)

            # Read the next line
            line = fp.readline()
        
        # Add final section to song
        if lines_in_section != 0:
            LpS_arr = np.append(LpS_arr,lines_in_section)         #add num lines for prev section
        if temp_WpL != np.empty(0):
            WpLpS_arr = np.append(WpLpS_arr, np.mean(temp_WpL))   #add average line length for prev sections
        else:
            num_sections -= 1;

        #Calculate Repetitions
        marked = [False for i in range(len(first_arr))]
        for i in range(len(first_arr)):
            if marked[i] == False:
                marked[i] = True
                found = False
                for j in range(i+1,len(first_arr)):
                    if first_arr[i] == first_arr[j]:
                        marked[j] = True
                        found = True
                        repeats += 1
                if found:
                    repeats += 1
            else:
                continue

        #Produce Song Statistics
        #print("OVERALL SONG STATISTICS")
        data_vec[index][0] = len(LpS_arr)           #num sections
        data_vec[index][1] = np.mean(LpS_arr)       #avg lines/section
        data_vec[index][2] = np.std(LpS_arr)        #std dev lines/section
        data_vec[index][3] = np.mean(WpLpS_arr)           #avg words/line****
        data_vec[index][4] = np.std(WpLpS_arr)           #std dev words/line****
        data_vec[index][5] = repeats/len(LpS_arr)           #repetition score
        index += 1

#NORMALIZE THE DATA, SEND TO TEXT FILE!!!
print(data_vec)
# Write-Overwrites 
out = open("out.txt","w")#write mode
out_arr = np.array_str(data_vec) 
out.write(out_arr) 
out.close()
