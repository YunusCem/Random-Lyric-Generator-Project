# Final Project for EECE 5644
## Created by Kerem Enhoş, Muralikrishna Shanmugasundaram, William Varner and Yunus Cem Yılmaz

## The repository is formed in the following fashion:

Extracted and Edited Lyrics file includes all the 100 songs that were used in our analysis.

Data Extraction and Clustering Code file includes all the code that we used to clean and combine lyrics, extract features from these lyrics and k-means cluster these extracted features.

Clustering Input and Results file contains the extracted features from the song along with the k-means clustering analysis results.

Song Generation Code file includes all the phonetic alphabet converter files, along with the code that we used to generate songs using these extracted lyrics and features with Markov Chains, RNN LSTM and LSTM (which is not used in the final analysis).

Sample Songs from Each Model file contains 10 sample songs generated from each model using the lyrics and features that were collected.

Generated Song Analysis file contains all the analysis of the generated songs and comparisons between models.

### We would like to thank the following people for their code:

Max Woolf [minimaxir](https://github.com/minimaxir) whose [text generating neural network](https://github.com/minimaxir/textgenrnn) we used.

Michael Phillips [mphilli](https://github.com/mphilli) whose [phonetic alphabet converter](https://github.com/mphilli/English-to-IPA) we modified.

Kemal Burcak Kaplan [kaplanbr](https://github.com/kaplanbr) whose [markov lyric generator](https://github.com/kaplanbr/Serdar-Ortac-Lyrics-Generator) we heavily drew from.

Mohammed Ma'amari [mamarih1](https://towardsdatascience.com/@mamarih1) whose [LSTM model](https://towardsdatascience.com/ai-generates-taylor-swifts-song-lyrics-6fd92a03ef7e) we also drew from (but did not end up using for the final project).
