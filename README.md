## Speech_Recognition

# Simple word recognition using dynamic time warping

A common problem in audio processing is speech recognition. In this post, we utilize dynamic time warping (DTW) to recognize a short phrase from a string of spoken words, also known as a "sentence" by humans. 

# DTW Overview

 We choose to use dynamic time warping for this problem because of its ability to identify words spoken at different speeds, along with its simplicity. In particular, DTW is very good at matching two similar time-series multi-dimensional signals, which may be non-linearly shifted or warped: 

#  Audio data preprocessing: Mel-frequency cepstrum coefficients (MFCCs)

# Spoken words are recorded as time-series data:s

 Create MFCC's
 Convert the data to mfcc
 Remove mean and normalize each column of MFCC 
 Word identification
