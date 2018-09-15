
# Simple word recognition using dynamic time warping

from dtw import dtw
import matplotlib.pyplot as plt
import numpy as np
import librosa
import IPython.display
from IPython.display import Image



#  Audio data preprocessing: Mel-frequency cepstrum coefficients (MFCCs)

# Spoken words are recorded as time-series data:

y, sr = librosa.load('audio_samples/chai_tea_latte_word.m4a')
plt.plot(y)
IPython.display.Audio(data=y, rate=sr)


mfcc = librosa.feature.mfcc(y, sr)
librosa.display.specshow(mfcc)


# ### Applying DTW to audio data



yTest, srTest = librosa.load('audio_samples/elias_mothers_milk_sentence.m4a')
IPython.display.Audio(data=yTest, rate=srTest)



# ###### Training data 





y1, sr1 = librosa.load('audio_samples/elias_mothers_milk_word.m4a')
y2, sr2 = librosa.load('audio_samples/chris_mothers_milk_word.m4a')
y3, sr3 = librosa.load('audio_samples/yaoquan_mothers_milk_word.m4a')
IPython.display.Audio(data=y1, rate=sr1)
IPython.display.Audio(data=y2, rate=sr2)
IPython.display.Audio(data=y3, rate=sr3)


# Create MFCC's




#Convert the data to mfcc:
mfcc1 = librosa.feature.mfcc(y1, sr1)
mfcc2 = librosa.feature.mfcc(y2, sr2)
mfcc3 = librosa.feature.mfcc(y3, sr3)
mfccTest = librosa.feature.mfcc(yTest,srTest)

# Remove mean and normalize each column of MFCC 
import copy
def preprocess_mfcc(mfcc):
        mfcc_cp = copy.deepcopy(mfcc)
        for i in xrange(mfcc.shape[1]):
            	mfcc_cp[:,i] = mfcc[:,i] - np.mean(mfcc[:,i])
            	mfcc_cp[:,i] = mfcc_cp[:,i]/np.max(np.abs(mfcc_cp[:,i]))
        return mfcc_cp

mfcc1 = preprocess_mfcc(mfcc1)
mfcc2 = preprocess_mfcc(mfcc2)
mfcc3 = preprocess_mfcc(mfcc3)
mfccTest = preprocess_mfcc(mfccTest)


# Word identification


window_size = mfcc1.shape[1]
dists = np.zeros(mfccTest.shape[1] - window_size)

for i in range(len(dists)):
    	mfcci = mfccTest[:,i:i+window_size]
    	dist1i = dtw(mfcc1.T, mfcci.T,dist = lambda x, y: np.exp(np.linalg.norm(x - y, ord=1)))[0]
    	dist2i = dtw(mfcc2.T, mfcci.T,dist = lambda x, y: np.exp(np.linalg.norm(x - y, ord=1)))[0]
    	dist3i = dtw(mfcc3.T, mfcci.T,dist = lambda x, y: np.exp(np.linalg.norm(x - y, ord=1)))[0]
    	dists[i] = (dist1i + dist2i + dist3i)/3
plt.plot(dists)


# select minimum distance window
word_match_idx = dists.argmin()
# convert MFCC to time domain
word_match_idx_bnds = np.array([word_match_idx,np.ceil(word_match_idx+window_size)])
samples_per_mfcc = 512
word_samp_bounds = (2/2) + (word_match_idx_bnds*samples_per_mfcc)
word = yTest[word_samp_bounds[0]:word_samp_bounds[1]]
IPython.display.Audio(data=word, rate=sr1)


yTest, srTest = librosa.load('audio_samples/chris_mothers_milk_sentence_fast.m4a')
IPython.display.Audio(data=yTest, rate=srTest)



mfccTest = librosa.feature.mfcc(yTest,srTest)
mfccTest = preprocess_mfcc(mfccTest)


window_size = mfcc1.shape[1]*(1/2.) # use 1/2 window size
dists = np.zeros(mfccTest.shape[1] - window_size)

for i in range(len(dists)):
    	mfcci = mfccTest[:,i:i+window_size]
    	dist1i = dtw(mfcc1.T, mfcci.T,dist=lambda x, y: np.exp(np.linalg.norm(x - y, ord=1)))[0]
    	dist2i = dtw(mfcc2.T, mfcci.T,dist=lambda x, y: np.exp(np.linalg.norm(x - y, ord=1)))[0]
    	dist3i = dtw(mfcc3.T, mfcci.T)[0]
    	dists[i] = (dist1i + dist2i + dist3i)/3
plt.plot(dists)


word_match_idx = dists.argmin()
word_match_idx_bnds = np.array([word_match_idx,np.ceil(word_match_idx+window_size)])
samples_per_mfcc = 512
word_samp_bounds = (2/2) + (word_match_idx_bnds*samples_per_mfcc)

word = yTest[word_samp_bounds[0]:word_samp_bounds[1]]


IPython.display.Audio(data=word, rate=sr1)


# Again, success! We used the same training data, and identified the target phrase, only this time the target phrase was spoken much faster. This truly shows off the warping ability of DTW. 


