#!/usr/bin/env python3

# Python module import
import os
import sys
import argparse
import math 
import random
import copy
import platform

import pickle
import numpy as np
import time as tM
import datetime as dT

import single_hoj_set as sh_set

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Load data 

def load_data( byte_object=False, entities=10, data_path="", data_object_path="", classes=61, _verbose=False, number_of_entries=168 ):

	# Variables for statistics
	computational_start_time = tM.time()

	_tmp_data = []
	_dir_counter = 0

	# If a byte object of the data is available load it. 
	if( byte_object is True ):
		print("------------------------------------------------------------------\n")
		print("LD :: Load byte objects.")
		_data, _dir_counter = load_pickles( data_object_path, "data")
	else: # Else, load the data from the data path.
		print("------------------------------------------------------------------\n")
		print("LD :: Load data from path.")
		_data, _dir_counter = load_data_from_path( data_path, classes, _verbose, number_of_entries )

	computational_end_time = tM.time()
	timeDiff = dT.timedelta(seconds=computational_end_time - computational_start_time)
	print("\nLD :: Data has been loaded in : "+str(timeDiff)+" s \n" );

	return _data, _dir_counter

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Load helper for data from a set ( no previous pickled objects )

def load_data_from_path( data_path="", _classes=61, _verbose=False, number_of_entries=168 ):

	_fileCounter = 0
	_dir_Counter = 0
	_data = []

	# Chit chat
	if( _verbose >= 1 ):
		print("\nLDFP :: Load data into memory.")

	# List the directorys in the path
	directories = os.listdir(str(data_path))

	if( _verbose >= 2 ):
		print("LDFP :: directories: ", directories)

	print("\n\n")

	# lade und tainiere jeden HoJ-Ordner im Trainingsverzeichnis
	for directory in directories:

		if( _verbose >= 3 ):
			print("\n")
			print("LDFP :: Directory: ",directory)

		# Build the data exact path
		exact_data_path = str(data_path) + directory;

		# Load the whole dataset
		_dir_Counter = _dir_Counter + 1
		_hoj_set, _number_of_files_in_set = get_data_from_subdirectory(exact_data_path, _classes, _fileCounter, _verbose, number_of_entries)
		_data.append( _hoj_set )

		_fileCounter = _fileCounter + _number_of_files_in_set
		# Print filecounter in one line
		sys.stdout.write("\r%d files have been loaded" % (_fileCounter) )
		sys.stdout.flush()

	sys.stdout.write(" from %d directories." % (_dir_Counter) )
	return _data, _dir_Counter

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Get the data of a single sequence 

def get_data_from_subdirectory( directory="empty", classes=61, fileCounter=0, verbose=False, number_of_entries=168 ):
	
	# A single label
	label = np.zeros(classes)
	
	# All labels of the set
	hoj_labels = []
	
	# The number of sucessful loaded files
	fileCounter = 0

	# List all files in the subdirectoy
	hoj_set_files = os.listdir(directory)

	# Get the number of records in this subdirectory
	number_of_set_files = len(hoj_set_files)
	
	# Write the files in the subdirectory to stdout
	if( verbose >= 2 ):
		print('GDFSD :: Files in set: ', number_of_set_files)

	# Create a hoj_set with the correct size ( number_of_set_files x descriptor length )
	hoj_set = np.zeros((number_of_set_files,number_of_entries))

	# Load each file and store the content in the hoj_set
	for hoj_file in hoj_set_files:

		# Increase the filecounter
		fileCounter+=1

		file = open(directory + "/" + hoj_file,'rb')

		# Load the actual data file
		hoj_array = np.array(np.load(file))
		file.close()

		# Reshape the structure and add it to the whole set database
		hoj_array = hoj_array.flatten()
		hoj_set[fileCounter-1] = hoj_array

		# lade Labels (test output)
		idx = int(directory[-3:])
		label[idx] = 1
		hoj_labels.append(label)

	h_set = sh_set.single_hoj_set()
	h_set.set_hoj_set(hoj_set)
	h_set.set_hoj_label(hoj_labels)
	h_set.set_hoj_set_name(directory.split("/")[-1])

	return h_set, fileCounter

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Load previously stored pickle data objects 

def load_pickles( _path, _object="data" ):

	_data = []
	_labels = []

	if _object is "data":
		with open( _path, 'rb') as f:

			if( platform.system() is 'Linux' ):
				_data = pickle.load(f)
			elif( platform.system() is 'Windows' ):
				_data = pickle.load(f, encoding='latin1')
			else:
				_data = pickle.load(f)

			_dir_counter = len(_data)
			number_of_files = 0
			for _obj in _data:
				number_of_files = number_of_files + len(_obj.get_hoj_set())

			print("\nLP :: "+str(number_of_files)+" files from "+str(_dir_counter)+" directories were loaded from "+str(_path))
	else:
		print("LP :: Unknown data object location.")

	return _data, _dir_counter

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Divide the dataset in testing and training using the given proportion or a subset training samples as given as list.

def devide_dataset( _data=[], _number_of_directories=0, _training_list=None, _proportion=None, _verbose=False ):

	_testing_data = []
	_training_data = []
	_testing_sets = []

	_training_labels = []
	_testing_labels = []

	random_set_numbers=[]

	# No proportion or training list is given. Everything is in the training.
	if _training_list is None and _proportion is None:
		print("DD :: No proportion or training list was given. All samples are stored as training samples.")
		for _obj in _data:
			
			_training_data.append(_obj)
			_training_labels.append( np.nonzero(_obj.get_hoj_label()[0])[0][0] )

	# Is a proportion is given
	if _training_list is None and _proportion is not None:

		# Get the proportion value for each part.
		training_part = int(_proportion.split("/")[0]);
		testing_part = int(_proportion.split("/")[1]);

		# Compute the number of testing datasets
		number_of_test_sets = int(math.floor( ( _number_of_directories * testing_part ) / 100 ))
		if( _verbose >= 1 ):
			print("DD :: Number_of_testsets: "+str(number_of_test_sets)+" from a total of "+str(_number_of_directories)+" sets.")

		# Build a list of random set numbers without duplicates.
		random_set_numbers = random.sample(range(_number_of_directories), number_of_test_sets)

		# Build the final testing and training lists.
		for k in range(0,len(_data)):
			# If k is not in the random_set_number list than k is a training sample.
			if k not in random_set_numbers:
				_training_data.append(_data[k])
				_training_labels.append( np.nonzero(_data[k].get_hoj_label()[0])[0][0] )
			else: # If k is in the list than k is a testing sample.
				_testing_data.append(_data[k])
				_testing_labels.append( np.nonzero(_data[k].get_hoj_label()[0])[0][0] )

		if( _verbose >= 2 ):
			print("DD :: # of training samples   : ", len(_training_data) )
			print("DD :: # training labels         : ", len(_training_labels) )
			print("DD :: # of testing samples    :", len(_testing_data) )
			print("DD :: # testing labels          : ", len( _testing_labels) )

	# If a training list is given
	elif _training_list is not None and _proportion is None:

		# Step trough all directories and load the data into training and testset depending on the training list.
		for _obj in _data:
			
			# Is part of the training list.
			if to_train( _training_list, _obj.get_hoj_set_name() ) is True:
				_training_data.append(_obj)
				_training_labels.append( np.nonzero(_obj.get_hoj_label()[0])[0][0])
			else: # Is not part of the training list.
				_testing_data.append(_obj)
				_testing_labels.append( np.nonzero(_obj.get_hoj_label()[0])[0][0])

		if( _verbose >= 4 ):
			print("DD :: # of training samples   : ", len(_training_data) )
			print("DD :: # training labels         : ", len(_training_labels) )
			print("DD :: # of testing samples    :", len(_testing_data) )
			print("DD :: # testing labels          : ", len( _testing_labels) )

	return _training_data, _training_labels, _testing_data, _testing_labels

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# A small function to skip actions which are not in a defined training_list

def to_train( training_list=None, _sequence_name_="" ):

	# Step through the list of valid sequences for the training
	for key in training_list:
		# It's a valid training sequence if the sequence matches a key in the training list
		if( key in _sequence_name_ ):
			# If the action of the skeleton file is in the training_list.
			return True

	return False
