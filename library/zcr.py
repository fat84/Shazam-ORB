# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 22:13:24 2018

@author: Grenceng
"""
import numpy as np

def zero_crossing_rate(wavedata, block_length, sample_rate):
    
    #JUMLAH BLOK YANG AKAN DIPROSES
    num_blocks = int(np.ceil(len(wavedata)/block_length))
    
    #WAKTU BLOK TERSEBUT DIMULAI
    timestamps = (np.arange(0,num_blocks - 1) * (block_length / float(sample_rate)))
    
    zcr = []
    
    for i in range(0,num_blocks-1):
        
        start = i * block_length
        stop  = np.min([(start + block_length - 1), len(wavedata)])
        
        zc = 0.5 * np.mean(np.abs(np.diff(np.sign(wavedata[start:stop]))))
        zcr.append(zc)
    
    return np.asarray(zcr), np.asarray(timestamps)