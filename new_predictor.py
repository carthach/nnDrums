#!/usr/bin/python
import sys
import essentia
from essentia.standard import *
from essentia.standard import YamlOutput, PoolAggregator
from essentia import Pool, array
import os, glob, re
import numpy as np
import cv2

import matplotlib.pyplot as plt

from scikits.audiolab import Format, Sndfile, play

labels = ['kick', 'snare', 'hat']
noHeader = True

import os.path
import glob
import fnmatch

counter = 0

np.set_printoptions(threshold='nan')

def segment(audio):        
#        od1 = OnsetDetection(method = 'hfc')
        od2 = OnsetDetection(method = 'complex')

        # let's also get the other algorithms we will need, and a pool to store the results

        w = Windowing(type = 'hann')
        fft = FFT() # this gives us a complex FFT
        c2p = CartesianToPolar() # and this turns it into a pair (magnitude, phase)

        pool = essentia.Pool()

        # let's get down to business

        frameSize = 1024
        hopSize = 512

        for fstart in range(0, len(audio)-frameSize, hopSize):
                frame = audio[fstart:fstart+frameSize]
        
        for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512):
                mag, phase, = c2p(fft(w(frame)))
#                pool.add('features.hfc', od1(mag, phase))
                pool.add('features.complex', od2(mag, phase))


        # Phase 2: compute the actual onsets locations
        onsets = Onsets()

        # onsets_hfc = onsets(# this algo expects a matrix, not a vector
        #         array([ pool['features.hfc'] ]),

        #         # you need to specify weights, but as there is only a single
        #         # function, it doesn't actually matter which weight you give it
        #         [ 1 ])
#        np.savetxt(outFile, onsets_hfc, fmt='%f')

        #Let's just take the complex as an example
        onsets_complex = onsets(array([ pool['features.complex'] ]), [ 1 ])
#        onsets_hfc = onsets(array([ pool['features.hfc'] ]), [ 1 ])

        startTimes = onsets_complex
        endTimes = onsets_complex[1:]
        duration = Duration()
        endTimes = np.append(endTimes, duration(audio))

        slicer = Slicer(startTimes = array(startTimes), endTimes = array(endTimes))
        
        frames = slicer(audio)        

        lengthInFrames = 0
        for i in range(len(frames)):
                lengthInFrames = lengthInFrames + len(frames[i])

        format = Format('wav')
        global counter
        f = Sndfile('out'+ str(counter) + '.wav' , 'w', format, 1, 44100)
        counter = counter + 1
        f.write_frames(np.asarray(frames[0]))

        onsets = []

        for i in range(len(frames)):
                onset = dict()
                onset['time'] = startTimes[i]
                onset['frames'] = frames[i]

                end = len(onset['frames'])
                start = end - 44 * 10

                #Cleanup the end of the samples
                for j in range(start, end):
                        onset['frames'][j] = 0.0

                onsets.append(onset)

        return onsets

def extractFeatures(onsets):
        w = Windowing(type = 'hann')
        spectrum = Spectrum()
        mfcc = MFCC()
        erbbands = ERBBands()

        sampleRate = 44100
        halfSampleRate = sampleRate*0.5
        frameSize = 1024
        hopSize = 512

        centroid = Centroid(range=halfSampleRate)
        cm = CentralMoments(range=halfSampleRate)
        distShape = DistributionShape()

        pool = essentia.Pool()        

        for onsetIndex in range(len(onsets)):
                onset = onsets[onsetIndex]
                audio = onset['frames']

                spectrum.reset()
                mfcc.reset()
                cm.reset()
                distShape.reset()

                mfccs = []
                
                for fstart in range(0, len(audio)-frameSize, hopSize):
                        frame = audio[fstart:fstart+frameSize]
                        spec = spectrum(w(frame))
                        
                        mfcc_bands, mfcc_coeffs = mfcc(spec)
                        mfccs.append(mfcc_coeffs)

                        pool.add('lowlevel.mfcc', mfcc_coeffs)
                        pool.add('lowlevel.spectral_centroid', centroid(spec))
                        pool.add('lowlevel.erbbands', erbbands(spec))
                        #pool.add('lowlevel.mfcc_bands', mfcc_bands)

                        moments = cm(spec)
                        spread, skewness, kurtosis = dist = distShape(moments)

                        pool.add('lowlevel.spectral_spread', spread)
                        pool.add('lowlevel.spectral_skewness', skewness)
                        pool.add('lowlevel.spectral_kurtosis', kurtosis)

                aggrPool = PoolAggregator(defaultStats = [ 'mean', 'var' ])(pool)

                #Global computers
                zcr = ZeroCrossingRate()
                lat = LogAttackTime()

                envelope = Envelope()
                tct = TCToTotal()

                aggrPool.add('lowlevel.zcr', zcr(audio))
                aggrPool.add('lowlevel.log_attack_time', lat(audio))
                aggrPool.add('lowlevel.tct', tct(envelope(audio)))

                bassERB = np.mean(aggrPool['lowlevel.erbbands.mean'][0:7])
                midERB =  np.mean(aggrPool['lowlevel.erbbands.mean'][7:28])
                highERB =  np.mean(aggrPool['lowlevel.erbbands.mean'][28:])

                aggrPool.add('lowlevel.erbbands.bass', bassERB)
                aggrPool.add('lowlevel.erbbands.mid', midERB)
                aggrPool.add('lowlevel.erbbands.high', highERB)

                features = np.concatenate([
                        aggrPool['lowlevel.zcr'],
                        aggrPool['lowlevel.tct'],
                        aggrPool['lowlevel.log_attack_time'],
                        [aggrPool['lowlevel.spectral_centroid.mean']],
                        [aggrPool['lowlevel.spectral_spread.mean']],
                        [aggrPool['lowlevel.spectral_skewness.mean']],
                        [aggrPool['lowlevel.spectral_kurtosis.mean']],
                        aggrPool['lowlevel.erbbands.bass'],
                        aggrPool['lowlevel.erbbands.mid'],
                        aggrPool['lowlevel.erbbands.high']])

                features = np.concatenate([features, aggrPool['lowlevel.mfcc.mean']])

                del onset['frames']

                onset['features'] = features

        return onsets

def splitBands(filename):
        loader = essentia.standard.MonoLoader(filename = filename)

        # and then we actually perform the loading:
        audio = loader()

        lowPass = BandPass(cutoffFrequency=20, bandwidth=20)
        lowPassAudio = lowPass(audio)

        midPass = BandPass(cutoffFrequency=1000, bandwidth=250)
        midPassAudio = midPass(audio)
        
        hiPass = HighPass(cutoffFrequency=9000)
        hiPassAudio = hiPass(audio)
        
        audio = dict()
        audio['low'] = lowPassAudio
        audio['mid'] = midPassAudio
        audio['hi'] = hiPassAudio

        return audio

def parser():
	import argparse

	p = argparse.ArgumentParser()
        
        p.add_argument('input',help='files to be processed')

        # parse arguments
	args = p.parse_args()
	# print arguments
	# if args.verbose:
	# 	print args
	# return args
	return args
        
def main():
        # parse arguments
        args = parser()

        #Split to lo, mid, hi
        audio = splitBands(args.input)

        #Write out streams
        # for key in audio:
        #         f = Sndfile(key + '.wav', 'w', Format('wav'), 1, 44100)
        #         f.write_frames(np.asarray(audio[key]))
        #         f.close()
                
        #Separate into onsets
        onsets = dict()
        for key in audio:
                onsets[key] = segment(audio[key])
                
        #Get features for every onsets
        features = dict()
        for key in onsets:
                features[key] = extractFeatures(onsets[key])

        #Load up ANN
        nnet = cv2.ANN_MLP()
        nnet.load('model.xml')
        
        #We'll store the onset times here         
        times = dict()

        #indices corresponding to streams
        featureIDs = {
                'low' : 0,
                'mid' : 1,
                'hi' : 2
        }

        for key in features:
                times[key] = []
                for i in range(len(features[key])):
                        predictionsIn = np.asarray([features[key][i]['features']])
                        predictionsOut = -1 * np.ones((len(predictionsIn), 3), 'float')
                        nnet.predict(predictionsIn, predictionsOut)

                        predictionsOut = np.around(predictionsOut)
                        if predictionsOut[0][featureIDs[key]] == 1:
                                times[key].append(features[key][i]['time'])

        print times
                                             

if __name__ == '__main__':
	main()
