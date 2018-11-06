# -*- coding: utf-8 -*-
"""
Created on Sun Aug 05 13:36:55 2018

@author: Grenceng
"""

import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import pylab as pl
from scipy.signal import get_window
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'library/'))
import stft
import math
import zcr
import energy

INT16_FAC = (2**15)-1
INT32_FAC = (2**31)-1
INT64_FAC = (2**63)-1
norm_fact = {'int16':INT16_FAC, 'int32':INT32_FAC, 
             'int64':INT64_FAC,'float32':1.0,'float64':1.0}

#STEP 1 : TAKING FILES
file = "wavs/Ginada_Emaneman_WeDarling.wav"
chunk = 2048
rates, audio = wavfile.read(file)
wf = wave.open(file, "rb")
p = pyaudio.PyAudio()
stream = p.open(
    format = p.get_format_from_width(wf.getsampwidth()),
    channels = wf.getnchannels(),
    rate = wf.getframerate(),
    output = True
)

#STEP 2 : PLAY FILES (optional)
data = wf.readframes(chunk)
while data != '':
    stream.write(data)
    data = wf.readframes(chunk)

#STEP 3 : GET MAGNITUDE and PHASE
#N = 1024         #FFT length
M = int(rates*0.02)         #Analysis window size
#H = int(0.5*M)        #Overlap between window
#w = get_window("hamming", M)
audio = np.float32(audio)/norm_fact[audio.dtype.name]
#maxplotfreq = rates/8.82
#mX, pX = stft.stftAnal(audio, rates, w, N, H)
#numFrames = int(mX[1700:1800,0].size)
#frmTime = H*np.arange(numFrames)/float(rates)
#binFreq = rates*np.arange(N*maxplotfreq/rates)/N
#plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:int(N*maxplotfreq/rates+1)]))
#plt.close()

#STEP 4 : EXTRACT FREQUENCY FROM MAGNITUDE(mX)
#freqaxis = rates*np.arange(N/2)/float(N)
#loc = []
#for m in mX[:,:-1]:
#    loc.append(np.argmax(m))
#Freq = freqaxis[loc]
sminduration = 0.5  #seconds (bisa diganti)
smin = int(rates/float(M)*sminduration)

#STEP 5 : DEFINE ENERGY OF FRAME
E, e_timestamp = energy.root_mean_square(audio, M, rates)
plt.plot(e_timestamp, E)
plt.show()

Eaverage = np.average(E)
Etconst = 0.124
Ethreshold = Etconst * Eaverage
Esilences = np.array([], dtype=np.int32)
counter = []
for x in range(E.size):
    if E[x] < Ethreshold:
        counter.append(x)
        if len(counter) >= smin and x == E.size - 1:
            Esilences = np.append(Esilences, [counter[0], counter[-1]])
    else:
        if len(counter) >= smin:
            Esilences = np.append(Esilences, [counter[0], counter[-1]])
        counter = []
Esilences = np.reshape(Esilences,(-1,2))

#STEP 6 : COUNTING ZERO CROSSING RATE OF AUDIO
Z, z_timestamp = zcr.zero_crossing_rate(audio, M, rates)
plt.plot(z_timestamp, Z)
plt.show()

Zthreshold = 0.07
Zsilences = np.array([], dtype=np.int32)
counter = []
for i in range(Z.size):
    if Z[i] > Zthreshold:
        counter.append(i)
    else:
        if len(counter) >= smin:
            Zsilences = np.append(Zsilences, [counter[0], counter[-1]])
        counter = []
Zsilences = np.reshape(Zsilences,(-1,2))

#CUTING 2 SECONDS BEFORE SILENCES AUDIO
twosec = int(2*(rates/float(M)))
cuts = np.array([], dtype=np.int32)
for i in range(Esilences.shape[0]):
    cutpoint = Esilences[i,0]
    if cutpoint - twosec >= 0:
        cuts = np.append(cuts, audio[(cutpoint - twosec) * M: cutpoint * M])
