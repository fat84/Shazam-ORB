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
        self.rates, self.audio = wavfile.read(file)
        self.wf = wave.open(file, "rb")
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format = self.p.get_format_from_width(self.wf.getsampwidth()),
            channels = self.wf.getnchannels(),
            rate = self.wf.getframerate(),
            output = True
        )
        
    def play(self):
        data = self.wf.readframes(chunk)
        while data != '':
            self.stream.write(data)
            data = self.wf.readframes(chunk)
    
    def create_plot(self):
        plt.plot(np.arange(self.audio.size)/float(self.rates), self.audio)
        plt.axis([0, self.audio.size/float(self.rates), min(self.audio), max(self.audio)])
        plt.savefig("plots/time-domain.png", dpi=1200)
        plt.show()
        
        N = 512         #FFT rate
        M = 401         #Analysis window
        H = int(0.75*M)        #Overlapping
        w = get_window("hanning", M)
        self.audio = np.float32(self.audio)/norm_fact[self.audio.dtype.name]
        maxplotfreq = self.rates/8.82
        mX, pX = stft.stftAnal(self.audio, self.rates, w, N, H)
        numFrames = int(mX[:,0].size)
        frmTime = H*np.arange(numFrames)/float(self.rates)
        binFreq = self.rates*np.arange(N*maxplotfreq/self.rates)/N
        plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:int(N*maxplotfreq/self.rates+1)]))
        plt.axis("off")
        plt.savefig("plots/magnitude spectogram.png", dpi=1200)
        plt.show()
        
    def close(self):
        self.stream.close()
        self.p.terminate()

a = AudioFile("wavs/cello-double-cuted.wav")
a.play()
a.create_plot()
a.close()