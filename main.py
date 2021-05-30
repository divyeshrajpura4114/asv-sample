#!/usr/bin/env python3

import os
import sys
import h5py
import random
import argparse
import numpy as np
from random import shuffle
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def GetArgs():
	parser = argparse.ArgumentParser(description = "Automatic Speaker Verification", 
		formatter_class = argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--train-data-dir', type = str, dest = "train_data_dir", default = None, required = True, 
		help = 'Proivde valid input train data directory')
	parser.add_argument('--val-data-dir', type = str, dest = "val_data_dir", default = None, required = True, 
		help = 'Proivde valid input val data directory')
	sys.stderr.write(' '.join(sys.argv) + "\n")
	args = parser.parse_args()
	args = CheckArgs(args)
	return args

def CheckArgs(args):
	if not os.path.exists(args.train_data_dir):
		raise Exception(sys.argv[0] + ": " + "The script expects" + " " + args.train_data_dir + " " + "to exist")
	if not os.path.exists(args.val_data_dir):
		raise Exception(sys.argv[0] + ": " + "The script expects" + " " + args.val_data_dir + " " + "to exist")
	return args

def get_featsscp_dict(featsscp_path):
	featsscp_dict = {}
	with open(featsscp_path, "r") as f_featsscp:
		for utt_data in f_featsscp:
			utt_id, utt_feats_path = utt_data.split(" ")			
			featsscp_dict[utt_id] = utt_feats_path
	return featsscp_dict

def read_hdf5(utt_id, hdf5_path):
	hdf5_path = hdf5_path.replace("\n","")
	with h5py.File(hdf5_path, "r") as f_hdf5:
		utt_feats = f_hdf5[utt_id]["feats"][()]
	return utt_feats

def slice_feats(feats, win_size):
	# Normalizing before slicing
	num_frames, num_feats = feats.shape
	half_win_size = int(win_size / 2)

	sliced_feats = []
	while num_frames <= win_size:
		feats = np.append(feats, feats[:num_frames, :], axis = 0)
		num_frames, num_feats = feats.shape

	j = random.randrange(half_win_size, num_frames - half_win_size)
	if not j:
		sliced_feats = np.zeros(num_frames, 40, 'float64')
		sliced_feats[0:(feats.shape)[0]] = feats.shape
	else:
		sliced_feats = feats[j - half_win_size:j + half_win_size]
	return np.array(sliced_feats)

class ConvolutionalNeuralNetwork(nn.Module):
	def __init__(self, in_channels, out_features, hidden_size=64):
		super(ConvolutionalNeuralNetwork, self).__init__()
		self.in_channels = in_channels
		self.out_features = out_features
		self.hidden_size = hidden_size

		self.conv1 = self.conv3x3(in_channels, hidden_size)
		self.conv2 = self.conv3x3(hidden_size, hidden_size)

		self.classifier = nn.Linear(4800, out_features)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv2(x)
		features = x.view((x.size(0), -1))
		logits = self.classifier(features)
		return logits

	def conv3x3(self, in_channels, out_channels):
		return nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels, momentum=1., track_running_stats=False),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)

class SpkRecTrain(nn.Module):
	def __init__(self, featsscp_path, win_size):
		super(SpkRecTrain, self).__init__()
		self.win_size = win_size
		self.featsscp_path = featsscp_path

		self.featsscp_dict = get_featsscp_dict(self.featsscp_path)
		self.spks = list(sorted(set([x.split("-")[0] for x in list(self.featsscp_dict.keys())])))
		self.spkid2label_dict = {}
		for spk_index, spk_id in enumerate(self.spks):
			self.spkid2label_dict[spk_id] = spk_index

	def __len__(self):
		#return len(self.featsscp_dict)
		return 320

	def __getitem__(self, index):
		utt_id = list(self.featsscp_dict.keys())[index]
		spk_id = utt_id.split("-")[0]
		utt_path = self.featsscp_dict[utt_id]
		utt_feats = read_hdf5(utt_id, utt_path)
		utt_feats = slice_feats(utt_feats, self.win_size)
		utt_feats = utt_feats.T
		spk_label = self.spkid2label_dict[spk_id]
		return utt_feats, spk_label

def train_epoch(epoch):
	model.train().to(device)
	
	losses = 0

	for batch_id, (batch_utt_feats, batch_spk_labels) in enumerate(train_loader):

		batch_utt_feats = batch_utt_feats.unsqueeze(1).to(torch.float).to(device)
		batch_spk_labels = batch_spk_labels.unsqueeze(1).to(torch.long).to(device)

		batch_utt_out = model(batch_utt_feats)
		batch_loss = objective(batch_utt_out, batch_spk_labels.squeeze(1))
		losses = losses + batch_loss.item()
		
		optimizer.zero_grad()
		batch_loss.backward()
		optimizer.step()

	losses = losses/len(train_loader)
	return losses

def val_epoch(epoch):
	model.eval()
	
	losses = 0

	for batch_id, (batch_utt_feats, batch_spk_labels) in enumerate(val_loader):
		batch_utt_feats = batch_utt_feats.unsqueeze(1).to(torch.float).to(device)
		batch_spk_labels = batch_spk_labels.unsqueeze(1).to(torch.long).to(device)

		batch_utt_out = model(batch_utt_feats)
		batch_loss = objective(batch_utt_out, batch_spk_labels.squeeze(1))
		losses = losses + batch_loss.item()
		

	losses = losses/len(val_loader)
	return losses

def do_training():
	train_losses = []
	val_losses = []

	start_epoch = 0
	end_epoch = num_epochs

	for epoch in range(start_epoch, end_epoch):
		train_epoch_loss = train_epoch(epoch)
		train_losses.append(train_epoch_loss)
	
		val_epoch_loss = val_epoch(epoch)
		val_losses.append(val_epoch_loss)

if __name__ == '__main__':

	args = GetArgs()

	device = "cpu" #"cuda:7"
	gpu_id = 0
	eps = 1e-6
	batch_size = 32
	lr = 0.01
	num_epochs = 10
	num_classes = 40

	train_featsscp_path = os.path.join(args.train_data_dir, "feats.scp")
	val_featsscp_path = os.path.join(args.val_data_dir, "feats.scp")
	
	model = ConvolutionalNeuralNetwork(1, num_classes).to(device)
	optimizer = optim.SGD([{'params': model.parameters(), 'weight_decay': 1e-4}], lr = lr, momentum = 0.9, nesterov = True, dampening = 0)
	objective = torch.nn.CrossEntropyLoss()

	train_data = SpkRecTrain(train_featsscp_path, 200)
	train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, drop_last = True)

	val_data = SpkRecTrain(val_featsscp_path, 200)
	val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = True, drop_last = True)

	do_training()
