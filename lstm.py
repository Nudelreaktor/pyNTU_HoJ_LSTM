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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

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
def lstm_init():

	# ----------------------------------------------------------------------------------------------------------------------------------------------------------
	# Initial operations.
	# ----------------------------------------------------------------------------------------------------------------------------------------------------------

	# Parse the command line options.
	clpDict = parseOpts( sys.argv ) 

	# Create base filename for output.
	statistics_base_filename = ""
	if clpDict['_output_name'] is not None:
		statistics_base_filename = clpDict['_output_name']
	else:
		statistics_base_filename = str(timestamp + "_" + "lstm" + "_c" + str(clpDict['_lstm_classes']) + "_e" + str(clpDict['_lstm_epochs']) + "_" + "-".join(str(x) for x in clpDict['_hid_layer_size']))

	# Check if clf_statistics folder is existing.
	if not os.path.exists("clf_statistics/"):
		os.makedirs("clf_statistics/")

	# Check if classifiers folder is existing.
	if not os.path.exists("classifiers/"):
		os.makedirs("classifiers/")

	# Build the classifier and train it.
	print("Init :: Creating neural network.")

	# ----------------------------------------------------------------------------------------------------------------------------------------------------------
	# Build the model, train it and test it.
	# ----------------------------------------------------------------------------------------------------------------------------------------------------------

	# Store the start time for statistics.
	start_time = time.time()

	# Create the neural network ( 2 Layer LSTM )
	model = Sequential()

	# Add LSTM layers to the model.
	if(len(clpDict['_hid_layer_size']) == 1):
		model.add(\
			LSTM( int(clpDict['_hid_layer_size'][0]),\
			input_shape=(None,clpDict['_in_layer_size'])\
			))
	else:
		for i in range(len(clpDict['_hid_layer_size'])):
			if i == 0:
				model.add(\
					LSTM( int(clpDict['_hid_layer_size'][i]),\
					input_shape=(None, clpDict['_in_layer_size']),\
					return_sequences=True,\
					activation=clpDict['_activation'],\
					recurrent_activation=clpDict['_recurrent_activation']\
					))
			else:
				if i == len(clpDict['_hid_layer_size']) - 1:
					model.add(\
						LSTM(int(clpDict['_hid_layer_size'][i]),\
						activation=clpDict['_activation'],\
						recurrent_activation=clpDict['_recurrent_activation']\
						))
				else:
					model.add(\
						LSTM(int(clpDict['_hid_layer_size'][i]),\
						return_sequences=True,\
						activation=clpDict['_activation'],\
						recurrent_activation=clpDict['_recurrent_activation']\
						))


	# Build fully connected layers for the dimension reduction of the high dimensional output data from the previuos layer.
	model.add(Dense(clpDict['_lstm_classes']))
	
	# Use softmax activation function. ( https://en.wikipedia.org/wiki/Softmax_function )
	model.add(Activation('softmax'))

	# Define the learning rate.	( Controlled artificial cooling. )
	optimizer = RMSprop(lr=0.001)

	# Define the loss function. ( Categorical_crossentropy -> one output is 1 all the others are 0 )
	model.compile(\
		loss='categorical_crossentropy',\
		optimizer=optimizer,\
		metrics=['accuracy']\
		)

	model.summary()
		
	# Load  dataset.
	if(os.path.isfile(clpDict['_data_object_path'])):
		dataset, dataset_size = dr.load_data(
									byte_object=True,\
									data_object_path=clpDict['_data_object_path'],\
									classes=clpDict['_lstm_classes'],\
									number_of_entries=clpDict['_in_layer_size']\
									)
	else:
		dataset, dataset_size = dr.load_data(\
									byte_object=False,\
									data_path=clpDict['_data_object_path'],\
									number_of_entries=clpDict['_in_layer_size']\
									)

	# Devide the dataset in training an testing data.
	training_dataset, _ , validation_dataset, _ = dr.devide_dataset(\
													_data=dataset,\
													_number_of_directories=dataset_size,\
													_training_list=clpDict['_training_list'],\
													_proportion=clpDict['_proportion'])

	# Train the lstm network.
	model, histories = lstm_train(\
							model,\
							training_dataset,\
							epochs=clpDict['_lstm_epochs'],\
							subsampling=clpDict['_subsampling'],\
							number_of_subframes=clpDict['_number_of_subframes'],\
							sample_strategy=clpDict['_sample_strategy'],\
							batch_size=clpDict['_batch_size'])
	
	# Validate the lstm network.
	score, acc, cnf_matrix = lstm_validate(\
							model,\
							validation_dataset,\
							create_confusion_matrix=True,\
							number_of_subframes=clpDict['_number_of_subframes'],\
							sample_strategy=clpDict['_sample_strategy'],\
							batch_size=clpDict['_batch_size'])

	# Store the end time for statistics.
	end_time = time.time()

	# Compute computational time.
	timeDiff = datetime.timedelta(seconds=end_time - start_time)

	# ----------------------------------------------------------------------------------------------------------------------------------------------------------
	# Print and write statistics
	# ----------------------------------------------------------------------------------------------------------------------------------------------------------

	# Build the filename for the stats file.
	clfStats_filename = "clf_statistics/" + statistics_base_filename + ".clfStats"
	check_Path(clfStats_filename)
	write_stats_file(\
		clfStats_filename,\
		timeDiff, histories,\
		acc,\
		score)

	# Build the filename for the confusion matrix file.
	cnfMatrix_filename = "clf_statistics/" + statistics_base_filename + ".cnfMatrix"
	# If the file doesn't exist, create it and store the matrix file.
	check_Path(cnfMatrix_filename)
	file = open(cnfMatrix_filename, "wt")
	writer = csv.writer(file)
	writer.writerows(cnf_matrix)

	# Build the filename for the confusion matrix file.
	cnfBMP_filename = "clf_statistics/" + statistics_base_filename + "_cnfMatrix.bmp"
	# Create a bitmap image of confusion matrix.
	img = create_bitmap( cnf_matrix )
	check_Path(cnfBMP_filename)
	img.save(cnfBMP_filename)

	# bonus bonus create .png image with matplotlib 
	cnfPNG_filename = "clf_statistics/" + statistics_base_filename + "_cnfMatrix.png"
	check_Path(cnfPNG_filename)
	store_conf_matrix_as_png(\
		cnf_matrix,\
		cnfPNG_filename)
	
	# save neural network
	if( clpDict['_save_net'] is True ):
		# If a output name was defined.
		if clpDict['_output_name'] is not None:
			filename = "classifiers/"+clpDict['_output_name']+".h5"
			check_Path(filename)
			model.save(filename)
		# else use the standard format.
		else:
			filename = "classifiers/" + filename_base + ".h5"
			check_Path(filename)
			model.save(filename)
	return model

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Create a bitmap from the confussion matrix.

def write_stats_file( clfStats_filename, timeDiff, histories, acc, score ):

	# Open the stats file.
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

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Create a bitmap from the confussion matrix.

def create_bitmap( cnf_matrix ):
	img = Image.new('RGB',(len(cnf_matrix) * 10,len(cnf_matrix) * 10),"black")
	pixels = img.load()

	for i in range(img.size[0]):
		for j in range (img.size[1]):
			if int(i/10) == int(j/10):
				pixels[i,j] = (0,int(cnf_matrix[int(j/10),int(i/10)] * 255),0)
			else:
				pixels[i,j] = (int(cnf_matrix[int(j/10),int(i/10)] * 255),0,0)

	return img

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Check if path exist

def check_Path( _path ):
	if not os.path.exists(os.path.dirname( _path )):
		os.makedirs(os.path.dirname( _path ))

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Store the confusion matrix as .png

def store_conf_matrix_as_png( cnf_matrix, _classifier_name ):

	import plot_confusion_matrix as p_CM 

	print("SCMP :: CLF_Name_ ", _classifier_name )
	cm_labels = np.arange(len(cnf_matrix[0]))
	p_CM.plot_confusion_matrix( cnf_matrix, cm_labels, _classifier_name , normalize=False, show=False )
	
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

def lstm_train(lstm_model, training_dataset, epochs=10, subsampling=False, number_of_subframes=8, sample_strategy="random", batch_size=32):
	
	print("train neural network...")
	histories = []

	# Trainingsepochen
	for x in range(0, epochs):
		print("Epoch", x+1, "/", epochs)
		# lade und tainiere jeden HoJ-Ordner im Trainingsverzeichnis
		training_data = []
		training_labels = []
		idx = 0

		for _obj in training_dataset:
			if subsampling is True:
				training_data.append(get_buckets(_obj.get_hoj_set(), number_of_subframes, sample_strategy))
			else:
				training_data.append(_obj.get_hoj_set())
			training_labels.append(_obj.get_hoj_label()[0]) 

		print(len(training_data[0]))
		print(len(training_labels[0]))

		# train neural network
		training_history = lstm_model.fit(np.array(training_data), np.array(training_labels), epochs=1, batch_size=batch_size, verbose=1) # epochen 1, weil ausserhald abgehandelt
		histories.append(training_history.history)
			
	return lstm_model, histories

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Validate the neural network

def lstm_validate(lstm_model, evaluation_dataset, create_confusion_matrix=False, number_of_subframes=0, sample_strategy="random", batch_size=32):
	
	print("evaluate neural network...")
	validation_data = []
	validation_labels = []
	
	accuracy = 0
	n = 0
	idx = 0

	
	for _obj in evaluation_dataset:
		if number_of_subframes > 0:
			validation_data.append(get_buckets(_obj.get_hoj_set(), number_of_subframes, sample_strategy))
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

		norm = Normalizer()
		cnf_matrix = norm.fit_transform(cnf_matrix)

		return score, acc, cnf_matrix


	return score, acc, None

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Evaluate data with the neural network.

def lstm_predict(lstm_model, hoj3d_set):
	prediction = lstm_model.predict(hoj3d_set,batch_size = 1)
	idx = np.argmax(prediction)[0]
	return idx,prediction[0][0][idx],prediction
	
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Skip actions during training which are not in the clpDict['_training_list']

def to_train( _training_list, _skeleton_filename_ ):
	# If an clpDict['_training_list'] is given 
	if( _training_list is not None ):
		for key in _training_list:
			if( key in _skeleton_filename_ ):
				# If the action of the skeleton file is in the clpDict['_training_list'].
				return True
	# If no clpDict['_training_list'] is given
	else:
		return True

	# If the action of the skeleton file is not in the clpDict['_training_list'].
	return False
	
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Skip actions during evaluation which are in the clpDict['_training_list']

def to_evaluate( _training_list, _skeleton_filename_ ):
	# If an clpDict['_training_list'] is given 
	if( c_training_list is not None ):
		for key in _training_list:
			if( key in _skeleton_filename_ ):
				# If the action of the skeleton file is in the clpDict['_training_list'].
				return False

	# If the action of the skeleton file is not in the clpDict['_training_list'].
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

	# Storage parameters
	_data_object_path = None
	_output_name = "empty"

	# Dataset control parameters 
	_training_list = None
	_subsampling = False
	_sample_strategy = "first"
	_number_of_subframes = 8
	_proportion = None

	# Classifier control parameters
	_activation = 'tanh'
	_recurrent_activation = 'tanh'
	_hid_layer_size = [16,16]
	_batch_size = 1
	_in_layer_size = 2
	_lstm_classes = 61
	_lstm_epochs = 10

	# general control parameters
	_save_net = False
	_verbose = False

	# Dictionary
	clpDict={\
		'_data_object_path':_data_object_path,\
		'_output_name':_output_name,\
		'_training_list':_training_list,\
		'_subsampling':_subsampling,\
		'_sample_strategy':_sample_strategy,\
		'_number_of_subframes':_number_of_subframes,\
		'_proportion':_proportion,\
		'_activation':_activation,\
		'_recurrent_activation':_recurrent_activation,\
		'_hid_layer_size':_hid_layer_size,\
		'_batch_size':_batch_size,\
		'_in_layer_size':_in_layer_size,\
		'_lstm_classes':_lstm_classes,\
		'_lstm_epochs':_lstm_epochs,\
		'_save_net':_save_net,\
		'_verbose':_verbose,\
	}

	# generate parser object
	parser = argparse.ArgumentParser()

	# Storage parameters
	parser.add_argument("-dop", "--data_object_path", action='store', dest="data_object_path", help="The path to the data_object. (required or -tp)")
	parser.add_argument("-oN", "--output_name", action='store', dest='output_name', help="The name which we will be used for the statistic and the classifier.") 

	# Dataset control parameters 
	parser.add_argument("-tl", "--training_list", action='store', dest='training_list', help="A list of training feature in the form: -tl S001,S002,S003,... (overrites -pp)")
	parser.add_argument("-sub","--subsampling", action='store_true', dest="subsampling", help="Will you subsample the dataset?")
	parser.add_argument("-ss", "--sample_strategy", action='store', dest='sample_strategy', help="Defines the strategy of the set subsampling. [ first ( default ) | mid | last | random]")
	parser.add_argument("-sf", "--number_of_subframes", action='store', dest="number_of_subframes", help="The number of frames per set in the training.")
	parser.add_argument("-pp", "--proportion", action='store', dest='proportion', help="The Proportion of the Datasets training data to validation data in the form -p 80/20")

	# Classifier control parameters
	parser.add_argument("-a", "--activation", action='store', dest='activation', help="The Activation function of the LSTM neurons. (default tanh)")
	parser.add_argument("-ra","--recurrent_activation", action='store', dest='recurrent_activation', help="the recurrent update function of the internal memory state. (default tanh)")
	parser.add_argument("-ls","--hidden_layer_size", action='store', dest='hid_layer_size', help="A list of sizes of the LSTM layers (standart: -ls 16,16)")
	parser.add_argument("-b", "--batch_size", action='store', dest='batch_size', help="The batch size to train the LSTM with.")
	parser.add_argument("-s", "--input_size", action='store', dest="in_layer_size", help="The number of input fields. (required)")
	parser.add_argument("-c", "--classes", action='store', dest="lstm_classes", help="The number of output classes. (required)")
	parser.add_argument("-e", "--epochs", action='store', dest="lstm_epochs", help="The number of training epochs. (required)")

	# general control parameters
 	parser.add_argument("-sn", "--save_network", action='store_true', dest='save_network', help="If set the created neural network will be saved.")
 	parser.add_argument("-v", "--verbose", action='store', dest='verbose', help="Get the chit chat.")

	# finally parse the command line 
	args = parser.parse_args()

	# ----------------------------------------------------------------------------------------------------------------------------------------------------------
	# Storage parameters
	# ----------------------------------------------------------------------------------------------------------------------------------------------------------

	# The path to the data_object.
	if args.data_object_path:
		clpDict['_data_object_path'] = args.data_object_path

	# The name which we will be used for the statistic and the classifier.
	if args.output_name:
		clpDict['_output_name'] = args.output_name

	# ----------------------------------------------------------------------------------------------------------------------------------------------------------
	# Dataset control parameters
	# ----------------------------------------------------------------------------------------------------------------------------------------------------------

	# A list of data objects for the training
	if args.training_list:
		clpDict['_training_list'] = args.training_list.split(",")

	# The number of subframes u will use from each subset.
	if args.subsampling:
		clpDict['_subsampling'] = args.subsampling

	# The number of subframes u will use from each subset.
	if args.number_of_subframes:
		clpDict['_number_of_subframes'] = int(args.number_of_subframes)

	# Which strategy will u use for the set subsampling?
	if args.sample_strategy: 
		clpDict['_sample_strategy'] = args.sample_strategy

	# Will u use a proportion for splitting the data in training and testing?
	if args.proportion:
		clpDict['_proportion'] = args.proportion

	# ----------------------------------------------------------------------------------------------------------------------------------------------------------
	# Classifier control parameters
	# ---------------------------------------------------------------------------------------------------------------------------------------------------------- 

	# Specify the activation function for the lstm cells.
	if args.activation:
		clpDict['_activation'] = args.activation

	# Specify the recurrent activation function for the lstm cells. ( Backprop path )
	if args.recurrent_activation:
		clpDict['_recurrent_activation'] = args.recurrent_activation

	# Define the layer size for the lstm layers.
	if args.hid_layer_size:
		clpDict['_hid_layer_size'] = args.hid_layer_size.split(",")

	# Define the batch size for the trianing. ( Online (0) vs. batch learning (n, n > 0). )
	if args.batch_size:
		clpDict['_batch_size'] = int(args.batch_size)

	# The size of the hidden layer.
	if args.in_layer_size:
		clpDict['_in_layer_size'] = int(args.in_layer_size)

	# The number of training clpDict['_lstm_epochs'].
	if args.lstm_epochs:
		clpDict['_lstm_epochs'] = int(args.lstm_epochs)

	# The number of clpDict['_lstm_classes'] in the computation.
	if args.lstm_classes:
		clpDict['_lstm_classes'] = int(args.lstm_classes)

	# Will you save the network after training
	if args.save_network is True:
		clpDict['_save_net'] = True

	# ----------------------------------------------------------------------------------------------------------------------------------------------------------
	# Global control parameters
	# ----------------------------------------------------------------------------------------------------------------------------------------------------------        

	if args.verbose:
		clpDict['_verbose'] = int(args.verbose)

	# ----------------------------------------------------------------------------------------------------------------------------------------------------------
	# Commandline Parameter Output
	# ----------------------------------------------------------------------------------------------------------------------------------------------------------  

	print ("\nControl Configuration:")
	print ("-----------------------------------------------------------------")
	print ("Input size         : ", clpDict['_in_layer_size'])
	print ("Output size        : ", clpDict['_lstm_classes'])
	print ("Training epochs    : ", clpDict['_lstm_epochs'])
	print ("Lstm destination   : ", clpDict['_output_name'])
	if args.save_network is True:
		print("Network will be saved")		
	else:
		print("Network won't be saved!")
	if clpDict['_verbose'] > 0:
		print ("Verbosity level           : ", clpDict['_verbose']  )

	return clpDict

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
	lstm_init()