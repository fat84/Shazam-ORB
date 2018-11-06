# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 18:42:57 2018

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

INT16_FAC = (2**15)-1
INT32_FAC = (2**31)-1
INT64_FAC = (2**63)-1
norm_fact = {'int16':INT16_FAC, 'int32':INT32_FAC, 'int64':INT64_FAC,'float32':1.0,'float64':1.0}
class AudioFile:
    global chunk
    chunk = 1024
    def __init__(self, file):
        global audio
        global rates
        self.rates, self.audio = wavfile.read(file)
        self.wf = wave.open(file, "rb")
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format = self.p.get_format_from_width(self.wf.getsampwidth()),
            channels = self.wf.getnchannels(),
            rate = self.wf.getframerate(),
            output = True
        )
        audio = self.audio
        rates = self.rates
        
    def play(self):
        data = self.wf.readframes(chunk)
        while data != '':
            self.stream.write(data)
            data = self.wf.readframes(chunk)
        
        
    
    def create_plot(self):
        p = np.arange(self.audio.size)/float(self.rates)
        duration = p[-1]
        plt_len = duration*0.5
        plt.plot(np.arange(self.audio.size)/float(self.rates), self.audio)
        plt.axis([0, self.audio.size/float(self.rates), min(self.audio), max(self.audio)])
        plt.savefig("plots/time-domain.png")
        plt.close()
        
        N = 8192         #FFT
        M = 8192         #Analysis window size
        H = int(0.75*M)        #Overlap between window
        w = get_window("hamming", M)
        self.audio = np.float32(self.audio)/norm_fact[self.audio.dtype.name]
        maxplotfreq = self.rates/8.82
        mX, pX = stft.stftAnal(self.audio, self.rates, w, N, H)
        numFrames = int(mX[:,0].size)
        frmTime = H*np.arange(numFrames)/float(self.rates)
        binFreq = self.rates*np.arange(N*maxplotfreq/self.rates)/N
        plt.figure(figsize=(plt_len,1))
        plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:int(N*maxplotfreq/self.rates+1)]))
        plt.axis("off")
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                            hspace = 0, wspace = 0)
        plt.savefig("plots/magnitude spectogram.png", dpi=300)
        plt.close()
        
        plt.plot(mX)
        plt.axis("off")
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                            hspace = 0, wspace = 0)
        plt.savefig("plots/magnitude plot.png")
        plt.close()
        global mx
        mx = mX
    def close(self):
        self.stream.close()
        self.p.terminate()

a = AudioFile("wavs/b0272.wav")
a.play()
a.create_plot()
a.close()