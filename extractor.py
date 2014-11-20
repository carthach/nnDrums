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

def extractOnsets(audio):
        od1 = OnsetDetection(method = 'hfc')
        od2 = OnsetDetection(method = 'complex')

        # let's also get the other algorithms we will need, and a pool to store the results

        w = Windowing(type = 'hann')
        fft = FFT() # this gives us a complex FFT
        c2p = CartesianToPolar() # and this turns it into a pair (magnitude, phase)

        pool = essentia.Pool()

        # let's get down to business
        for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512):
                mag, phase, = c2p(fft(w(frame)))
                pool.add('features.hfc', od1(mag, phase))
                pool.add('features.complex', od2(mag, phase))


        # Phase 2: compute the actual onsets locations
        onsets = Onsets()

        onsets_hfc = onsets(# this algo expects a matrix, not a vector
                array([ pool['features.hfc'] ]),

                # you need to specify weights, but as there is only a single
                # function, it doesn't actually matter which weight you give it
                [ 1 ])
#        np.savetxt(outFile, onsets_hfc, fmt='%f')

        #Let's just take the complex as an example
        onsets_complex = onsets(array([ pool['features.complex'] ]), [ 1 ])

        startTimes = onsets_hfc
        endTimes = onsets_hfc[1:]
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

        return frames


        
        
def extractFeaturesFromOnset(samples, outFile, outFileAggr, label=None):
	sampleRate = 44100

	w = Windowing(type = 'hann')
	spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
	mfcc = MFCC()
	erbbands = ERBBands()
	
	mfccs = []

	# for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512):
	# 	mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
	# 	mfccs.append(mfcc_coeffs)

	pool = essentia.Pool()

	# or as a one-liner:
	YamlOutput(filename = outFile)(pool)

	# and ouput those results in a file

	halfSampleRate = sampleRate*0.5

	centroid = Centroid(range=halfSampleRate)
	cm = CentralMoments(range=halfSampleRate)
	distShape = DistributionShape()
	
	for frame in FrameGenerator(samples, frameSize = 1024, hopSize = 512):
		spec = spectrum(w(frame))
		mfcc_bands, mfcc_coeffs = mfcc(spec)

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
	
	aggrPool.add('lowlevel.zcr', zcr(samples))
	aggrPool.add('lowlevel.log_attack_time', lat(samples))
	aggrPool.add('lowlevel.tct', tct(envelope(samples)))

	bassERB = np.mean(aggrPool['lowlevel.erbbands.mean'][0:7])
	midERB =  np.mean(aggrPool['lowlevel.erbbands.mean'][7:28])
	highERB =  np.mean(aggrPool['lowlevel.erbbands.mean'][28:])

	aggrPool.add('lowlevel.erbbands.bass', bassERB)
	aggrPool.add('lowlevel.erbbands.mid', midERB)
	aggrPool.add('lowlevel.erbbands.high', highERB)

	# x = np.concatenate(classVector, np.array(aggrPool['lowlevel.tct']))
	x = np.concatenate([
                aggrPool['lowlevel.zcr'],
		aggrPool['lowlevel.tct'],
		aggrPool['lowlevel.log_attack_time'],
		[aggrPool['lowlevel.spectral_centroid.mean']],
		[aggrPool['lowlevel.spectral_spread.mean']],
		[aggrPool['lowlevel.spectral_skewness.mean']],
		[aggrPool['lowlevel.spectral_kurtosis.mean']],
		aggrPool['lowlevel.erbbands.bass'],
		aggrPool['lowlevel.erbbands.mid'],
		aggrPool['lowlevel.erbbands.high']]
		)

        x = np.concatenate([x, aggrPool['lowlevel.mfcc.mean']])

        if label is not None:
                x = np.concatenate([x, [label]])

	# np.savetxt('/Users/carthach/Desktop/instances.txt', instances, fmt='%f')

	YamlOutput(filename = outFileAggr)(aggrPool)

	return x

def extractInstances(inputFiles, csvFilename='', labelled=False, type=''):
        #If samples are one shot we just populate instaces directly with the extracted audio
        #If samples are full we extract an array of instances for every samples and append 
	files = []
	for f in inputFiles:
		if os.path.isdir(f):
			for root, dirnames, filenames in os.walk(f):
				for filename in fnmatch.filter(filenames, '*.wav'):
					files.append(os.path.join(root, filename))
		else:
			# file was given, append to list
			files.append(f)

	# only process .wav files
	files = fnmatch.filter(files, '*.wav')
	files.sort()

        kickPattern = re.compile('BD_')                        
        snarePattern = re.compile('SD_')
        hatPattern = re.compile('HH_')

        instances = []
        for f in files:
                filename = os.path.splitext(f)[0]
                outFile = "%s.txt" % (filename)
                aggrOutFile = "%s.aggr.txt" % (filename)

                label = None
                if labelled:
                        if kickPattern.search(f):
                                label = 0
                                print "here"
                        elif snarePattern.search(f):
                                label = 1
                        elif hatPattern.search(f):
                                label = 2
                        else:
                                continue
                        
                
                # we start by instantiating the audio loader:
                loader = essentia.standard.MonoLoader(filename = f)

                # and then we actually perform the loading:
                audio = loader()

                onsets = [0]
                if type=='oneshot':
                        onsets[0] = audio
                else:
                        onsets = extractOnsets(audio)

                for i in range(len(onsets)):
                        instance = extractFeaturesFromOnset(onsets[i], outFile, aggrOutFile, label=label)
                        instances.append(instance)

        instances = np.asarray(instances)

        if csvFilename != '':
                csvHeader = "zcr, tct, lat, spectral_centroid, spectral_spread, spectral_skewness, spectral_kurtosis, erbbands.bass, erbbands.mid, erbbands.high"

                for i in range(0, 13):
                        csvHeader += ",mfcc" + str(i)

                csvHeader+= ",label"

                np.savetxt(csvFilename, instances, fmt='%f', delimiter = ',',header=csvHeader, comments="")

        return instances

def createANN(data,classes):
        ninputs = data.shape[1]
        nhidden = 4
        noutputs = 3
        layers = np.array([ninputs, nhidden,noutputs])
        nnet = cv2.ANN_MLP(layers, cv2.ANN_MLP_SIGMOID_SYM, 1,1)
#        nnet = cv2.ANN_MLP(layers)

        criteria = (
                cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                10000,
                0.001)

        params = dict(
                term_crit = criteria,
                train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP, 
                bp_dw_scale = 0.05, 
                bp_moment_scale = 0.05 )

        num_iter = nnet.train(data, classes, None, params= params)

        nnet.save('model.xml')

        return nnet

def roundPredictions(predictions):
        for i in range(len(predictions)):
                print predictions[i]
                current_one = -1
                current_two = -1
                index_one = 0
                index_two = 0
                for j in range(3):
                        if predictions[i][j] > current_one:
                                current_two = current_one
                                current_one = predictions[i][j]
                                index_one = index_two
                                index_two = j
                predictions[i][index_one] = 1
                predictions[i][index_two] = 1

        predictions = np.around(predictions)
        return predictions


def predict(nnet, data, classes=None):
        # Create a matrix of predictions
        predictions = np.zeros((len(data),3), 'float32')

        print 'data'
        print data

        np.savetxt('data.txt', data, fmt='%f')
        
        # See how the network did.
        nnet.predict(data, predictions)

#        predictions = np.around(predictions)
#        predictions = roundPredictions(predictions)

        print 'predictions:'
        print predictions
        
        if classes is not None:
                # Compute sum of squared errors
                sse = np.sum( (classes - predictions)**2 )

                # Compute # correct
                true_labels = np.argmax(classes, axis=0 )
                pred_labels = np.argmax( predictions, axis=0 )
                num_correct = np.sum( true_labels == pred_labels )

                print 'targets:'
                print classes

                print 'sum sq. err:', sse
                print 'accuracy:', float(num_correct) / len(true_labels)
        #        print 'ran for %d iterations' % num_iter
                # print 'inputs:'
                # print data

        return predictions

def outputPattern(predictions):
        pattern = []
        for i in range(len(predictions)):
                for j in range(3):
                        if predictions[i][j] == 1:
                                pattern.append(j)

        x = [0,0,1]
        y = [5,6,7]
        x = np.arange(len(pattern))
        plt.scatter(x, y)
        plt.show()

def parser():
	import argparse

	p = argparse.ArgumentParser()

        p.add_argument('-t', dest='train', action='store_true',
                       help='train')

        p.add_argument('-p', dest='predict', action='store_true',
                       help='predict')

        p.add_argument('-l', dest='labelled', action='store_true',
                       help='predict')

        p.add_argument('-o', dest='oneshot', action ='store_true',
                       help='predict')

        
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
        
        #Load input data (model or raw audio)
        if args.train:
                f = []
                f.append(args.input)

                training_source = extractInstances(f, type='oneshot', labelled=True)

                noOfColumns = len(training_source[0])

                training_data = np.float32(training_source[:,:noOfColumns-1])

                print(len(training_data[0]))
                training_classes = np.float32(training_source[:,noOfColumns-1])
                training_ann_classes =  -1 * np.ones((len(training_classes), 3), 'float')


                for i in range(0, len(training_classes)):
                        training_ann_classes[i][int(training_classes[i])] = 1

                print training_ann_classes                        

                nnet = createANN(training_data,training_ann_classes)
                
        #Predict
        if args.predict:
                if not os.path.isfile('model.xml'):
                        print "No model.xml file, are you sure you haven't trained the neural net?"
                        return
                nnet = cv2.ANN_MLP()
                nnet.load('model.xml')

                f = []
                f.append(args.input)

                type = ''
                if(args.oneshot):
                        type='oneshot'
                
                testing_source = extractInstances(f, type=type, labelled=args.labelled)
                noOfColumns = len(testing_source[0])

                if(args.labelled):
                        testing_data = np.float32(testing_source[:,:noOfColumns-1])
                        testing_classes = np.float32(testing_source[:,noOfColumns-1])
                        testing_ann_classes = -1 * np.ones((len(testing_classes), 3), 'float')

                        for i in range(0, len(testing_classes)):
                                testing_ann_classes[i][int(testing_classes[i])] = 1

                        predictions = predict(nnet, testing_data, testing_ann_classes)
#                        outputPattern(predictions)
                else:
                        testing_data = np.float32(testing_source[:,:noOfColumns])
                        predictions = predict(nnet, testing_data)
                        outputPattern(predictions)

if __name__ == '__main__':
	main()
