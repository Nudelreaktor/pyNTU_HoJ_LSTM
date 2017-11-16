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
import tkinter as tk
from tkinter import filedialog

# local
import dataset_reader as dr
import single_hoj_set as sh_set


timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')


def lstm_init(save = False):


	# Parse the command line options.
	save, lstm_path, epochs, classes, hoj_height, training_path, training_list, layer_sizes, dataset_pickle_path, sample_strategy = parseOpts( sys.argv )

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
				model.add(LSTM(int(layer_sizes[i]), input_shape=(None,hoj_height), return_sequences=True))
			else:
				if i == len(layer_sizes) - 1:
					model.add(LSTM(int(layer_sizes[i])))
				else:
					model.add(LSTM(int(layer_sizes[i]), return_sequences=True))


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

	model, histories = lstm_train(model, classes, epochs=epochs, training_directory=training_path, training_list=training_list, dataset_pickle_file=dataset_pickle_path, _sample_strategy=sample_strategy, number_of_entries=hoj_height)
	
	
	#	evaluation_path = training_path
	score, acc, cnf_matrix = lstm_validate(model, classes, evaluation_directory=training_path, training_list=training_list, dataset_pickle_file=dataset_pickle_path, create_confusion_matrix=True, number_of_entries=hoj_height)

	end_time = time.time()

	timeDiff = datetime.timedelta(seconds=end_time - start_time)

	# print statistics
	if not os.path.exists("clf_statistics/"):
		os.makedirs("clf_statistics/")
	filename = "clf_statistics/" + filename_base + ".clfStats"

	f = open(filename, "wt")	
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

	file = open("clf_statistics/" + filename_base + "_confusion_matrix.conf_matrix", "wt")
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


	img.save("clf_statistics/" + filename_base + "_confusion_matrix.bmp")
	
	
	# save neural network
	if save is True:
		if lstm_path is not None:
			model.save(lstm_path)
		else:
			if not os.path.exists("classifiers/"):
				os.makedirs("classifiers/")
			filename = "classifiers/" + filename_base + ".h5"
			model.save(filename)

	return model
	
	
# use this funktion to load a trained neural network
def lstm_load(filename = None):
	if filename is not None:
		return load_model(filename)

	f = filedialog.askopenfilename(filetypes=(("Model files","*.h5"),("all files","*.*")))
	if f is None or f is "":
		return None
	else:
		return load_model(f)

#use this funktion to train the neural network
def lstm_train(lstm_model, classes, epochs=10, training_directory="lstm_train/", training_list=None, dataset_pickle_file="", _sample_strategy="random", number_of_entries=168):
	
	print("train neural network...")
	directories = os.listdir(training_directory)
	directories_len = len(directories)

	complete_hoj_data = None
	histories = []



	# read dataset
	if(os.path.isfile(dataset_pickle_file)):
		dataset, dataset_size = dr.load_data(byte_object=True, data_object_path=dataset_pickle_file, classes=classes, number_of_entries=number_of_entries)
	else:
		dataset, dataset_size = dr.load_data(byte_object=False, data_path=training_directory, number_of_entries=number_of_entries)

	training_dataset = dr.devide_dataset(_data=dataset, _training_list=training_list)[0]

	# Trainingsepochen
	for x in range(0,epochs):
		print("Epoch", x+1, "/", epochs)
		# lade und tainiere jeden HoJ-Ordner im Trainingsverzeichnis
		training_data = []
		training_labels = []
		idx = 0

		for _obj in training_dataset:
			training_data.append(get_eight_buckets(_obj.get_hoj_set()))
			training_labels.append(_obj.get_hoj_label()[0])

		# train neural network
		training_history = lstm_model.fit(np.array(training_data), np.array(training_labels), epochs=1, batch_size=32, verbose=1) # epochen 1, weil außerhald abgehandelt; batch_size 1, weil data_sets unterschiedliche anzahl an Frames
		histories.append(training_history.history)
			
	return lstm_model, histories

#use this funktion to train the neural network
def lstm_validate(lstm_model, classes, evaluation_directory="lstm_train/", training_list=None, dataset_pickle_file="", create_confusion_matrix=False, _sample_strategy="random", number_of_entries=168):
	
	print("evaluate neural network...")
	directories = os.listdir(evaluation_directory)
	directories_len = len(directories)
	validation_data = []
	validation_labels = []
	
	accuracy = 0
	n = 0
	idx = 0

	# read dataset and labels
	
	if(os.path.isfile(dataset_pickle_file)):

		dataset, dataset_size = dr.load_data(byte_object=True, data_object_path=dataset_pickle_file, classes=classes, number_of_entries=number_of_entries)

	else:
		dataset, dataset_size = dr.load_data(byte_object=False, data_path=evaluation_directory, number_of_entries=number_of_entries)

	if training_list is not None:
		evaluation_dataset = dr.devide_dataset(_data=dataset, _training_list=training_list)[2]
	else:
		evaluation_dataset = dataset

	for _obj in evaluation_dataset:
		validation_data.append(get_eight_buckets(_obj.get_hoj_set()))
		validation_labels.append(_obj.get_hoj_label()[0])


	# evaluate neural network
	score, acc = lstm_model.evaluate(np.array(validation_data), np.array(validation_labels), batch_size=32, verbose=0) # batch_size willkuerlich
			
	print("Accuracy:",acc)

	if create_confusion_matrix is True:
		predictions = lstm_model.predict(np.array(validation_data),batch_size = 32)
		
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


def get_hoj_data(directory, classes):
	hoj_set_files = os.listdir(directory)
	data = []
	hoj_set = []
	label = np.zeros(classes)
	# alle dateien laden, in einer Matrix peichern
	for hoj_file in hoj_set_files:
		file = open(directory + "/" + hoj_file,'rb')
		hoj_array = np.load(file)
		file.close()

		hoj_set.append(hoj_array)

	# lade Labels (test output)
	idx = int(directory[-3:])
	label[idx - 1] = 1

	selected_hoj_set = get_eight_buckets(hoj_set)

	return np.array(selected_hoj_set), label


		
# use this funktion to evaluate data in the neural network
def lstm_predict(lstm_model, hoj3d_set):
	prediction = lstm_model.predict(hoj3d_set,batch_size = 1)
	idx = np.argmax(prediction)[0]
	return idx,prediction[0][0][idx],prediction
	
# ----------------------------------------------------------------------------------------------------------------------------------------------------------

# A small function to skip actions which are not in the training_list
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

# A small function to skip actions which are in the training_list
def to_evaluate( training_list, _skeleton_filename_ ):
	# If an training_list is given 
	if( training_list is not None ):
		for key in training_list:
			if( key in _skeleton_filename_ ):
				# If the action of the skeleton file is in the training_list.
				return False

	# If the action of the skeleton file is not in the training_list.
	return True


def get_eight_buckets( hoj_set, _sample_strategy="random" ):

	# Get some informations about the data
	number_of_frames = len(hoj_set)
	_number_of_subframes = 8
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

# Parse the command line arguments
def parseOpts( argv ):

	# generate parser object
	parser = argparse.ArgumentParser()
	# add arguments to the parser so he can parse the shit out of the command line

	# dataset parameters
	parser.add_argument("-dop", "--data_object_path", action='store', dest="data_object_path", help="The path to the data_object. (required or -tp)")
	parser.add_argument("-tp", "--training_path", action='store', dest="training_path", help="The path of the training directory. (required or -dp)")
	parser.add_argument("-tl", "--training_list", action='store', dest='training_list', help="A list of training feature in the form: -tl S001,S002,S003,...")

	# classifier parameters
	parser.add_argument("-s", "--input_size", action='store', dest="lstm_size", help="The number of input fields. (required)")
	parser.add_argument("-c", "--classes", action='store', dest="lstm_classes", help="The number of output classes. (required)")
	parser.add_argument("-e", "--epochs", action='store', dest="lstm_epochs", help="The number of training epochs. (required)")
	parser.add_argument("-ls", "--layer_sizes", action='store', dest='layer_sizes', help="A list of sizes of the LSTM layers (standart: -ls 16,16)")
	parser.add_argument("-bs", "--bucket_strategy", action='store', dest='bucket_strategy', help="Defines the strategy of the set subsampling. [first | mid | last | random]")

	# general control parameters
	parser.add_argument("-p", "--path", action='store', dest="lstm_path", help="The PATH where the lstm-model will be saved.")
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

	if args.layer_sizes:
		layer_sizes = args.layer_sizes.split(",")
	else:
		layer_sizes = [16,16]

	if args.data_object_path:
		data_object_path = args.data_object_path
	else:
		data_object_path = ""

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

	return (not args.test_network), lstm_path, lstm_epochs, lstm_classes, lstm_size, training_path, training_list, layer_sizes, data_object_path, args.bucket_strategy

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
	lstm_init(True)