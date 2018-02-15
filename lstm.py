#!/usr/bin/env python3

# Python module import
import os
import numpy as np
import math as ma
import argparse
import sys
import random
import time
import datetime
import json
import pickle
import csv

from PIL import Image

# keras import
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.layers import Dense, Activation, LSTM
from keras.preprocessing import sequence

# confusion matrix
from sklearn.metrics import confusion_matrix

# file dialog
# TODO Rebuild tKinter stuff with something else simple
# import tkinter as tk
# from tkinter import filedialog

# local
import dataset_reader as dr
import single_hoj_set as sh_set
import plot_confusion_matrix as p_CM 


timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# LSTM's main routine.
def lstm_init(save = False):


	# Parse the command line options.
	save, lstm_path, epochs, classes, hoj_height, training_path, training_list, layer_sizes, dataset_pickle_path, sample_strategy, number_of_subframes, batch_size, proportion, activation, recurrent_activation = parseOpts( sys.argv )

	filename_base = timestamp + "_" + "lstm" + "_c" + str(classes) + "_e" + str(epochs) + "_" + "-".join(str(x) for x in layer_sizes)

	print("creating neural network...")

	start_time = time.time()

	# create neural network
	# 2 Layer LSTM
	model = Sequential()

	# LSTM Schichten hinzufuegen
	if(len(layer_sizes) == 1):
		model.add(LSTM(int(layer_sizes[0]), input_shape=(None,hoj_height)))
	else:
		for i in range(len(layer_sizes)):
			if i == 0:
				model.add(LSTM(int(layer_sizes[i]), input_shape=(None,hoj_height), return_sequences=True, activation=activation, recurrent_activation=recurrent_activation))
			else:
				if i == len(layer_sizes) - 1:
					model.add(LSTM(int(layer_sizes[i]), activation=activation, recurrent_activation=recurrent_activation))
				else:
					model.add(LSTM(int(layer_sizes[i]), return_sequences=True, activation=activation, recurrent_activation=recurrent_activation))


	# voll vernetzte Schicht zum Herunterbrechen vorheriger Ausgabedaten auf die Menge der Klassen 
	model.add(Dense(classes))
	
	# Aktivierungsfunktion = Transferfunktion
	# Softmax -> hoechsten Wert hervorheben und ausgaben normalisieren
	model.add(Activation('softmax'))
	
	# lr = Learning rate
	# zur "Abkuehlung" des Netzwerkes
	optimizer = RMSprop(lr=0.001)
	# categorical_crossentropy -> ein Ausgang 1 der Rest 0
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	model.summary()
	
	
	# read dataset
	if(os.path.isfile(dataset_pickle_path)):
		dataset, dataset_size = dr.load_data(byte_object=True, data_object_path=dataset_pickle_path, classes=classes, number_of_entries=hoj_height)
	else:
		dataset, dataset_size = dr.load_data(byte_object=False, data_path=dataset_pickle_path, number_of_entries=hoj_height)

	training_dataset, _ , validation_dataset, _ = dr.devide_dataset(_data=dataset, _number_of_directories=dataset_size, _training_list=training_list, _proportion=proportion)

	model, histories = lstm_train(model, training_dataset, epochs=epochs, number_of_subframes=number_of_subframes, _sample_strategy=sample_strategy, batch_size=batch_size)
	
	
	#	evaluation_path = training_path
	score, acc, cnf_matrix = lstm_validate(model, validation_dataset, create_confusion_matrix=True,number_of_subframes=number_of_subframes, _sample_strategy=sample_strategy, batch_size=batch_size)

	end_time = time.time()

	timeDiff = datetime.timedelta(seconds=end_time - start_time)

	# print statistics
	
	# Create base filename 
	statistics_base_filename = ""
	if lstm_path is not None:
		statistics_base_filename = lstm_path
	else:
		statistics_base_filename = filename_base

	# Check if clf_statistics is existing
	if not os.path.exists("clf_statistics/"):
		os.makedirs("clf_statistics/")

	clfStats_filename = "clf_statistics/" + statistics_base_filename + ".clfStats"
	# If clf_statistics/ exist but there is no file 
	if not os.path.exists(os.path.dirname(clfStats_filename)):
		os.makedirs(os.path.dirname(clfStats_filename))

	f = open(clfStats_filename, "wt")	
	f.write("Network was created and trained in : "+str(timeDiff)+" s\n" )
	f.write("------------------------------------------------------------------\n")
	for x in range(0,len(histories)):

		f.write("Training epoch : "+str(x+1)+"\tloss: "+str(histories[x]['loss'])+"\tacc: "+str(histories[x]["acc"])+"\n")
	f.write("------------------------------------------------------------------\n")
	f.write("\n")
	f.write("Prediction Accuracy: "+str(acc))
	f.write("\n")
	f.write("Network score      : "+str(score))
	f.write("\n")
	f.write("------------------------------------------------------------------\n")
	f.close()
	print("network creation succesful! \\(^o^)/")

	# save confusion matrix 
	###############################################################
	#
	# confusion matrix:
	# vertical true label
	# horrizontal predicted label
	#
	###############################################################

	cnfMatrix_filename = "clf_statistics/" + statistics_base_filename + ".cnfMatrix"
	# If file not exist, create.
	if not os.path.exists(os.path.dirname(cnfMatrix_filename)):
		os.makedirs(os.path.dirname(cnfMatrix_filename))
	file = open(cnfMatrix_filename, "wt")
	writer = csv.writer(file)
	writer.writerows(cnf_matrix)

	# bonus create Bitmap image of confusion matrix
	img = Image.new('RGB',(len(cnf_matrix) * 10,len(cnf_matrix) * 10),"black")
	pixels = img.load()

	for i in range(img.size[0]):
		for j in range (img.size[1]):
			if int(i/10) == int(j/10):
				pixels[i,j] = (0,int(cnf_matrix[int(j/10),int(i/10)] * 255),0)
			else:
				pixels[i,j] = (int(cnf_matrix[int(j/10),int(i/10)] * 255),0,0)

	cnfBMP_filename = "clf_statistics/" + statistics_base_filename + "_cnfMatrix.bmp"
	img.save(cnfBMP_filename)

	# bonus bonus create .png image with matplotlib 
	cnfPNG_filename = "clf_statistics/" + statistics_base_filename + "_cnfMatrix.png"
	if not os.path.exists(os.path.dirname(cnfPNG_filename)):
		os.makedirs(os.path.dirname(cnfPNG_filename))
	store_conf_matrix_as_png( cnf_matrix, cnfPNG_filename )
	
	# save neural network
	if save is True:

		if not os.path.exists("classifiers/"):
			os.makedirs("classifiers/")

		if lstm_path is not None:
			filename = "classifiers/"+lstm_path+".h5"
			if not os.path.exists(os.path.dirname(filename)):
				os.makedirs(os.path.dirname(filename))
			model.save(filename)
		else:
			filename = "classifiers/" + filename_base + ".h5"
			if not os.path.exists(os.path.dirname(filename)):
				os.makedirs(os.path.dirname(filename))
			model.save(filename)

	return model

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Store the confusion matrix as .png

def store_conf_matrix_as_png( cnf_matrix, _classifier_name ):

	import plot_confusion_matrix as p_CM 

	print("SCMP :: CLF_Name_ ", _classifier_name )
	cm_labels = np.arange(len(cnf_matrix[0]))
	p_CM.plot_confusion_matrix( cnf_matrix, cm_labels, _classifier_name , True, show=False )
	
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Load a previously trained neural network.

def lstm_load(filename = None):
	if filename is not None:
		return load_model(filename)

# TODO Rebuild tKinter stuff with something else simple
	# f = filedialog.askopenfilename(filetypes=(("Model files","*.h5"),("all files","*.*")))
	# if f is None or f is "":
	# 	return None
	# else:
	# 	return load_model(f)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Train the neural network.

def lstm_train(lstm_model, training_dataset, epochs=10, number_of_subframes=8, _sample_strategy="random", batch_size=32):
	
	print("train neural network...")
	histories = []
	
	# Trainingsepochen
	for x in range(0,epochs):
		print("Epoch", x+1, "/", epochs)
		# lade und tainiere jeden HoJ-Ordner im Trainingsverzeichnis
		training_data = []
		training_labels = []
		idx = 0

		for _obj in training_dataset:
			if number_of_subframes > 0:
				training_data.append(get_buckets(_obj.get_hoj_set(), number_of_subframes, _sample_strategy))
			else:
				training_data.append(_obj.get_hoj_set())
			training_labels.append(_obj.get_hoj_label()[0])

		# train neural network
		training_history = lstm_model.fit(np.array(training_data), np.array(training_labels), epochs=1, batch_size=batch_size, verbose=1) # epochen 1, weil ausserhald abgehandelt
		histories.append(training_history.history)
			
	return lstm_model, histories

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Validate the neural network

def lstm_validate(lstm_model, evaluation_dataset, create_confusion_matrix=False, number_of_subframes=0, _sample_strategy="random", batch_size=32):
	
	print("evaluate neural network...")
	validation_data = []
	validation_labels = []
	
	accuracy = 0
	n = 0
	idx = 0

	
	for _obj in evaluation_dataset:
		if number_of_subframes > 0:
			validation_data.append(get_buckets(_obj.get_hoj_set(), number_of_subframes, _sample_strategy))
		else:
			validation_data.append(_obj.get_hoj_set())
		validation_labels.append(_obj.get_hoj_label()[0])


	# evaluate neural network
	score, acc = lstm_model.evaluate(np.array(validation_data), np.array(validation_labels), batch_size=batch_size, verbose=0)
			
	print("Accuracy:",acc)

	if create_confusion_matrix is True:
		predictions = lstm_model.predict(np.array(validation_data),batch_size = batch_size)
		
		predicted_labels = []
		real_labels = []

		for k in range(len(predictions)):
			predicted_idx = np.argmax(predictions[k])

			label_idx = np.argmax(validation_labels[k])
			
			real_labels.append(label_idx)
			predicted_labels.append(predicted_idx)


		cnf_matrix = confusion_matrix(real_labels, predicted_labels)

		cnf_matrix = cnf_matrix.astype('float') / (cnf_matrix.sum(axis=1) if cnf_matrix.sum(axis=1) is not 0 else 1) [:, np.newaxis]
		return score, acc, cnf_matrix


	return score, acc, None

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Evaluate data with the neural network.

def lstm_predict(lstm_model, hoj3d_set):
	prediction = lstm_model.predict(hoj3d_set,batch_size = 1)
	idx = np.argmax(prediction)[0]
	return idx,prediction[0][0][idx],prediction
	
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Skip actions during training which are not in the training_list

def to_train( training_list, _skeleton_filename_ ):
	# If an training_list is given 
	if( training_list is not None ):
		for key in training_list:
			if( key in _skeleton_filename_ ):
				# If the action of the skeleton file is in the training_list.
				return True
	# If no training_list is given
	else:
		return True

	# If the action of the skeleton file is not in the training_list.
	return False
	
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Skip actions during evaluation which are in the training_list

def to_evaluate( training_list, _skeleton_filename_ ):
	# If an training_list is given 
	if( training_list is not None ):
		for key in training_list:
			if( key in _skeleton_filename_ ):
				# If the action of the skeleton file is in the training_list.
				return False

	# If the action of the skeleton file is not in the training_list.
	return True

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Subsample the training data if necessary.

def get_buckets( hoj_set, _number_of_subframes, _sample_strategy="random" ):

	# Get some informations about the data
	number_of_frames = len(hoj_set)
	frame = []

	# Compute the size of the 8 buckets depending of the number of frames of the set.
	bucket_size = ma.floor( number_of_frames / _number_of_subframes )
	remain = number_of_frames - ( bucket_size * _number_of_subframes )
	gap = ma.floor(remain / 2.0)

	# Take a random frame from each bucket and store it as array entry in the _svm_structure ( 8 per )
	for k in range(0,_number_of_subframes):

		# Choose the sampling strategy
		# First frame per bucket
		if( _sample_strategy == "first"):
			random_frame_number = int(gap+(k*bucket_size)+1)
		# Mid frame per bucket
		elif( _sample_strategy == "mid"):
			random_frame_number = int(gap+(k*bucket_size)+int(ma.floor(bucket_size/2)))
		# Last frame per bucket
		elif( _sample_strategy == "last"):
			random_frame_number = int(gap+(k*bucket_size)+bucket_size)
		# Random frame per bucket
		else:
			# Get the random frame -> randint(k(BS),k+1(BS)) ==> k-1(B) < randomInt < k(B)
			random_frame_number = random.randint((gap+(k*bucket_size)),(gap+((k+1)*bucket_size)) )

		# Convert the frame to the svm structure 
		# Get the random frame and the corresponding label
		if( random_frame_number > 0 ):
			# Collect the data from the 8 buckets in a list.
			frame.append(hoj_set[random_frame_number-1]);
		else:
			# Collect the data from the 8 buckets in a list.
			frame.append(hoj_set[random_frame_number]);

	return frame
	
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Parse command line arguments

def parseOpts( argv ):

	# generate parser object
	parser = argparse.ArgumentParser()
	# add arguments to the parser so he can parse the shit out of the command line

	# dataset parameters
	parser.add_argument("-dop", "--data_object_path", action='store', dest="data_object_path", help="The path to the data_object. (required or -tp)")
	parser.add_argument("-tp", "--training_path", action='store', dest="training_path", help="The path of the training directory. (required or -dp)")
	parser.add_argument("-tl", "--training_list", action='store', dest='training_list', help="A list of training feature in the form: -tl S001,S002,S003,... (overrites -pp)")
	parser.add_argument("-pp", "--proportion", action='store', dest='proportion', help="The Proportion of the Datasets training data to validation data in the form -p 80/20")

	# classifier parameters
	parser.add_argument("-s", "--input_size", action='store', dest="lstm_size", help="The number of input fields. (required)")
	parser.add_argument("-c", "--classes", action='store', dest="lstm_classes", help="The number of output classes. (required)")
	parser.add_argument("-e", "--epochs", action='store', dest="lstm_epochs", help="The number of training epochs. (required)")
	parser.add_argument("-ls", "--layer_sizes", action='store', dest='layer_sizes', help="A list of sizes of the LSTM layers (standart: -ls 16,16)")
	parser.add_argument("-sf", "--number_of_subframes", action='store', dest="number_of_subframes", help="The number of subframes in one bucket. No subsampling when <= 0")
	parser.add_argument("-bs", "--bucket_strategy", action='store', dest='bucket_strategy', help="Defines the strategy of the set subsampling. [first | mid | last | random]")
	parser.add_argument("-b", "--batch_size", action='store', dest='batch_size', help="The batch size to train the LSTM with.")
	parser.add_argument("-a", "--activation", action='store', dest='activation', help="The Activation function of the LSTM neurons. (default tanh)")
	parser.add_argument("-ra", "--recurrent_activation", action='store', dest='recurrent_activation', help="the recurrent update function of the internal memory state. (default tanh)")

	# general control parameters
	parser.add_argument("-p", "--path", action='store', dest="lstm_path", help="The PATH with filename where the lstm-model and statistics will be saved.")
	parser.add_argument("-t", "--test", action='store_true', dest='test_network', help="if set the created neural network won't be saved. (overrites -p)")

	

	# finally parse the command line 
	args = parser.parse_args()

	if args.lstm_path:
		lstm_path = args.lstm_path
	else:
		lstm_path = None

	if args.lstm_epochs:
		lstm_epochs = int(args.lstm_epochs)
	else:
		lstm_epochs = 10

	if args.lstm_classes:
		lstm_classes = int(args.lstm_classes)
	else:
		lstm_classes = 1

	if args.lstm_size:
		lstm_size = int(args.lstm_size)
	else:
		lstm_size = 2
	
	if args.training_path:
		training_path = args.training_path
	else:
		training_path = None
		
	if args.training_list:
		training_list = args.training_list.split(",")
	else:
		training_list = None
		
	if args.proportion and training_list is None:
		proportion = args.proportion
	else:
		proportion = None

	if args.layer_sizes:
		layer_sizes = args.layer_sizes.split(",")
	else:
		layer_sizes = [16,16]

	if args.data_object_path:
		data_object_path = args.data_object_path
	else:
		data_object_path = ""
		
	if args.number_of_subframes:
		number_of_subframes = int(args.number_of_subframes)
	else:
		number_of_subframes = 0
	
	if args.batch_size:
		batch_size = int(args.batch_size)
	else:
		batch_size = 1
		
	if args.activation:
		activation = args.activation
	else:
		activation = 'tanh'
		
	if args.recurrent_activation:
		recurrent_activation = args.recurrent_activation
	else:
		recurrent_activation = 'tanh'

	print ("\nConfiguration:")
	print ("-----------------------------------------------------------------")
	print ("Input size         : ", lstm_size)
	print ("Output classes     : ", lstm_classes)
	print ("Training Epochs    : ", lstm_epochs)
	print ("Lstm destination   : ", lstm_path)
	if args.test_network is True:
		print("Network won't be saved!")
	else:
		print("Network will be saved")

	return (not args.test_network), lstm_path, lstm_epochs, lstm_classes, lstm_size, training_path, training_list, layer_sizes, data_object_path, args.bucket_strategy, number_of_subframes, batch_size, proportion, activation, recurrent_activation

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
	lstm_init(True)