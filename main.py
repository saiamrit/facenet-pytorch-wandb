import datetime
import time
import csv
import argparse

import numpy as np
import os
import torch
import time
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

from torchsummary import summary

import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.nn.modules.distance import PairwiseDistance

from dataset.TripletLossDataset import TripletFaceDataset, get_dataloader
from losses.Tripletloss import TripletLoss
from models.init_models import set_model_architecture, set_model_gpu_mode
from models.init_optimizer import set_optimizer
from utils import ModelSaver, write_csv, save_last_checkpoint, init_log_just_created, init_final, write_final
from eval_metrics import evaluate, plot_roc

import warnings
warnings.filterwarnings('ignore')

import wandb
wandb.login()

parser = argparse.ArgumentParser(description="Training a Patch based FaceNet facial recognition model using Triplet Loss.")
parser.add_argument('--num-epochs', default=200, type=int, metavar='NE',
					help='number of epochs to train (default: 100)')

parser.add_argument('--num-train-triplets', default=1000, type=int, metavar='NTT',
					help='number of triplets for training (default: 10000)')

parser.add_argument('--num-valid-triplets', default=1000, type=int, metavar='NVT',
					help='number of triplets for vaidation (default: 10000)')

parser.add_argument('--batch-size', default=64, type=int, metavar='BS',
					help='batch size (default: 64)')

parser.add_argument('--num-workers', default=4, type=int, metavar='NW',
					help='number of workers (default: 4)')

parser.add_argument('--learning-rate', default=0.0008, type=float, metavar='LR',
					help='learning rate (default: 0.001)')

parser.add_argument('--margin', default=0.5, type=float, metavar='MG',
					help='margin (default: 0.5)')

parser.add_argument('--patch', default='patch_0', type=str, help='which patch to train on')

parser.add_argument('--train-root-dir', default='/home/saiamrit/face_align/celebahq_sample/', type=str,
					help='path to train root dir')

parser.add_argument('--valid-root-dir', default='/home/saiamrit/face_align/celebahq_sample/', type=str,
					help='path to valid root dir')


parser.add_argument('--model_architecture', type=str, default="resnet50", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", 
																												"inceptionresnetv2", "mobilenetv2"],
					help="The required model architecture for training: ('resnet18','resnet34', 'resnet50', \
																'resnet101', 'resnet152', 'inceptionresnetv2', 'mobilenetv2'), (default: 'resnet18')")

parser.add_argument('--pretrained', default=False, type=bool,
					help="Download a model pretrained on the ImageNet dataset (Default: False)"
					)

parser.add_argument('--embedding_dimension', default=512, type=int,
					help="Dimension of the embedding vector (default: 512)")

parser.add_argument('--optimizer', type=str, default="adagrad", choices=["sgd", "adagrad", "rmsprop", "adam"],
					help="Required optimizer for training the model: ('sgd','adagrad','rmsprop','adam'), (default: 'adagrad')")

parser.add_argument('--step-size', default=500, type=int, metavar='SZ',
					help='Decay learning rate schedules every --step-size (default: 200)')

parser.add_argument('--unfreeze', type=str, metavar='UF', default='',
					help='Provide an option for unfreezeing given layers')

# parser.add_argument('--freeze', type=str, metavar='F', default='',
#                     help='Provide an option for freezeing given layers')

parser.add_argument('--load-best', action='store_true')

parser.add_argument('--load-last', action='store_true')
		
parser.add_argument('--continue-step', action='store_true')

parser.add_argument('--train-all', action='store_true', help='Train all layers')

args = parser.parse_args()

modelsaver = ModelSaver()

def save_if_best(state, acc):
	modelsaver.save_if_best(acc, state)

def init_wandb_config():
	config = dict(
		epoch = args.num_epochs,
		batch_size = args.batch_size,
		learning_rate = args.learning_rate,
		dataset = 'CELEBA-FFHQ',
		architecture = args.model_architecture,
		optimizer = args.optimizer,
		loss = 'Triplet Loss',
		patch = args.patch,
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	)
	run = wandb.init(project='facenet_patches_sweeps', config=config)
	config = wandb.config


def forward_pass(imgs, model, batch_size):
	imgs = imgs.cuda()
	embeddings = model(imgs)

	# Split the embeddings into Anchor, Positive, and Negative embeddings
	anc_embeddings = embeddings[:batch_size]
	pos_embeddings = embeddings[batch_size: batch_size * 2]
	neg_embeddings = embeddings[batch_size * 2:]

	return anc_embeddings, pos_embeddings, neg_embeddings, model

def create_folder(path):
	if not os.path.isdir(path):
		os.mkdir(path) 

def train(model, optimizer, triploss, scheduler, epoch, dataloaders, data_size, phase = 'train'):
	
	l2_dist = PairwiseDistance(2)
	labels, distances = [], []
	triplet_loss_sum = 0.0
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	if phase == 'train':
		scheduler.step()
		if scheduler.last_epoch % scheduler.step_size == 0:
			print("LR decayed to:", ', '.join(map(str, scheduler.get_last_lr())))
		model.train()
	else:
		model.eval()
		
	wandb.watch(model, triploss, log='all')

	progress_bar = enumerate(tqdm(dataloaders[phase], desc = 'Training Step'))

	for batch_idx, batch_sample in progress_bar:

		anc_img = batch_sample['anc_img'].to(device)
		pos_img = batch_sample['pos_img'].to(device)
		neg_img = batch_sample['neg_img'].to(device)

		# pos_cls = batch_sample['pos_class'].to(device)
		# neg_cls = batch_sample['neg_class'].to(device)

		with torch.set_grad_enabled(phase == 'train'):

			# anc_embed, pos_embed and neg_embed are encoding(embedding) of image
			anc_embed, pos_embed, neg_embed = model(anc_img), model(pos_img), model(neg_img)

			# choose the semi hard negatives only for "training"
			pos_dist = l2_dist.forward(anc_embed, pos_embed)
			neg_dist = l2_dist.forward(anc_embed, neg_embed)

			all = (neg_dist - pos_dist < args.margin).cpu().numpy().flatten()
			if phase == 'train':
				hard_triplets = np.where(all == 1)
				if len(hard_triplets[0]) == 0:
					continue
			else:
				hard_triplets = np.where(all >= 0)

			anc_hard_embed = anc_embed[hard_triplets]
			pos_hard_embed = pos_embed[hard_triplets]
			neg_hard_embed = neg_embed[hard_triplets]

			anc_hard_img = anc_img[hard_triplets]
			pos_hard_img = pos_img[hard_triplets]
			neg_hard_img = neg_img[hard_triplets]

			# pos_hard_cls = pos_cls[hard_triplets]
			# neg_hard_cls = neg_cls[hard_triplets]

#             model.module.forward_classifier(anc_hard_img)
#             model.module.forward_classifier(pos_hard_img)
#             model.module.forward_classifier(neg_hard_img)

			triplet_loss = triploss.forward(anc_hard_embed, pos_hard_embed, neg_hard_embed)

			if phase == 'train':
				optimizer.zero_grad()
				triplet_loss.backward()
				optimizer.step()

			distances.append(pos_dist.data.cpu().numpy())
			labels.append(np.ones(pos_dist.size(0)))

			distances.append(neg_dist.data.cpu().numpy())
			labels.append(np.zeros(neg_dist.size(0)))

			triplet_loss_sum += triplet_loss.item()

	avg_triplet_loss = triplet_loss_sum / data_size[phase]
	labels = np.array([sublabel for label in labels for sublabel in label])
	distances = np.array([subdist for dist in distances for subdist in dist])

	tpr, fpr, precision, recall, accuracy, roc_auc, tar, far, best_distance = evaluate(distances, labels)
	
	wandb.log({"Epoch": epoch,
			   "Train Loss": avg_triplet_loss,
			   "Train Accuracy": np.mean(accuracy),
			   "Train Precision":np.mean(precision),
			   "Train Recall":np.mean(recall),
			   "Train ROC AUC":roc_auc
			  })
	
	print()
	print("---------------------------------------  Train Stats  ------------------------------------------"
			  "\n\nAccuracy: {:.4f}\tPrecision {:.4f}\tRecall {:.4f}\n"
			  "ROC Area Under Curve: {:.4f}\tBest distance threshold: {:.2f}\t"
			  "TAR: {:.4f} @ FAR: {:.4f}".format(
					np.mean(accuracy),
					np.mean(precision),
					np.mean(recall),
					roc_auc,
					best_distance,
					tar,
					far
				)
		)
	
	print('\n\n{} set - Triplet Loss       = {:.8f}'.format(phase, avg_triplet_loss))
	print('{} set - Accuracy           = {:.8f}'.format(phase, np.mean(accuracy)))
	print(110 * '-')
	print();print()

	time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	lr = '_'.join(map(str, scheduler.get_lr()))
	# layers = '+'.join(args.unfreeze.split(','))
	write_csv(f"log_sweep/patchwise_logs/{phase}_{args.patch}.csv",[time, epoch, np.mean(accuracy), np.mean(precision), np.mean(recall), avg_triplet_loss, roc_auc, args.batch_size, lr])

	if phase == 'valid':
		save_last_checkpoint({'epoch': epoch,
							  'state_dict': model.module.state_dict(),
							  'optimizer_state': optimizer.state_dict(),
							  'accuracy': np.mean(accuracy),
							  'loss': avg_triplet_loss
							  })
		save_if_best({'epoch': epoch,
					  'state_dict': model.module.state_dict(),
					  'optimizer_state': optimizer.state_dict(),
					  'accuracy': np.mean(accuracy),
					  'loss': avg_triplet_loss
					  })

	create_folder('./log_sweep/roc_plots/')
	create_folder('./log_sweep/roc_plots/train/')


	# if not os.path.exists('./log/roc_plots/'):
	# 	os.mkdir('./log/roc_plots/')
	# if not os.path.exists('./log/roc_plots/train/'):
	# 	os.mkdir('./log/roc_plots/train/')
	plot_roc(fpr, tpr, figure_name='./log_sweep/roc_plots/train/roc_train_epoch_{}.png'.format(epoch))
	# plot_roc(fpr, tpr, figure_name='./log/roc_train_epoch_{}.png'.format(epoch))

	return np.mean(accuracy), np.mean(precision), np.mean(recall), avg_triplet_loss, roc_auc


def valid(model, optimizer, triploss, scheduler, epoch, dataloaders, data_size, phase = 'valid'):
	
	l2_dist = PairwiseDistance(2)
	labels, distances = [], []
	triplet_loss_sum = 0.0
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	if phase == 'train':
		scheduler.step()
		if scheduler.last_epoch % scheduler.step_size == 0:
			print("LR decayed to:", ', '.join(map(str, scheduler.get_lr())))
		model.train()
	else:
		model.eval()

	progress_bar = enumerate(tqdm(dataloaders[phase], desc = 'Validation Step'))

	for batch_idx, batch_sample in progress_bar:

		anc_img = batch_sample['anc_img'].to(device)
		pos_img = batch_sample['pos_img'].to(device)
		neg_img = batch_sample['neg_img'].to(device)


		# pos_cls = batch_sample['pos_class'].to(device)
		# neg_cls = batch_sample['neg_class'].to(device)

		with torch.set_grad_enabled(phase == 'train'):

			# anc_embed, pos_embed and neg_embed are encoding(embedding) of image
			anc_embed, pos_embed, neg_embed = model(anc_img), model(pos_img), model(neg_img)

			# choose the semi hard negatives only for "training"
			pos_dist = l2_dist.forward(anc_embed, pos_embed)
			neg_dist = l2_dist.forward(anc_embed, neg_embed)

			all = (neg_dist - pos_dist < args.margin).cpu().numpy().flatten()
			if phase == 'train':
				hard_triplets = np.where(all == 1)
				if len(hard_triplets[0]) == 0:
					continue
			else:
				hard_triplets = np.where(all >= 0)

			anc_hard_embed = anc_embed[hard_triplets]
			pos_hard_embed = pos_embed[hard_triplets]
			neg_hard_embed = neg_embed[hard_triplets]

			anc_hard_img = anc_img[hard_triplets]
			pos_hard_img = pos_img[hard_triplets]
			neg_hard_img = neg_img[hard_triplets]

			# pos_hard_cls = pos_cls[hard_triplets]
			# neg_hard_cls = neg_cls[hard_triplets]

#             model.module.forward_classifier(anc_hard_img)
#             model.module.forward_classifier(pos_hard_img)
#             model.module.forward_classifier(neg_hard_img)

			triplet_loss = triploss.forward(anc_hard_embed, pos_hard_embed, neg_hard_embed)

			if phase == 'train':
				optimizer.zero_grad()
				triplet_loss.backward()
				optimizer.step()

			distances.append(pos_dist.data.cpu().numpy())
			labels.append(np.ones(pos_dist.size(0)))

			distances.append(neg_dist.data.cpu().numpy())
			labels.append(np.zeros(neg_dist.size(0)))

			triplet_loss_sum += triplet_loss.item()

	avg_triplet_loss = triplet_loss_sum / data_size[phase]
	labels = np.array([sublabel for label in labels for sublabel in label])
	distances = np.array([subdist for dist in distances for subdist in dist])

	tpr, fpr, precision, recall, accuracy, roc_auc, tar, far, best_distance = evaluate(distances, labels)
	
	wandb.log({"Epoch": epoch,
			   "Validation Loss": avg_triplet_loss,
			   "Validation Accuracy": np.mean(accuracy),
			   "Validation Precision":np.mean(precision),
			   "Validation Recall":np.mean(recall),
			   "Validation ROC AUC":roc_auc
			  })
	
	print()
	print("---------------------------------------  Valid Stats  ------------------------------------------"
			  "\n\nAccuracy: {:.4f}\tPrecision {:.4f}\tRecall {:.4f}\n"
			  "ROC Area Under Curve: {:.4f}\tBest distance threshold: {:.2f}\t"
			  "TAR: {:.4f} @ FAR: {:.4f}".format(
					np.mean(accuracy),
					np.mean(precision),
					np.mean(recall),
					roc_auc,
					best_distance,
					tar,
					far
				)
		)
	
	print('\n\n{} set - Triplet Loss       = {:.8f}'.format(phase, avg_triplet_loss))
	print('{} set - Accuracy           = {:.8f}'.format(phase, np.mean(accuracy)))

	time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	lr = '_'.join(map(str, scheduler.get_lr()))
	layers = '+'.join(args.unfreeze.split(','))
	write_csv(f'log_sweep/patchwise_logs/{phase}_{args.patch}.csv', [time, epoch, np.mean(accuracy), np.mean(precision), np.mean(recall), avg_triplet_loss, roc_auc, args.batch_size, lr])

	if phase == 'valid':
		save_last_checkpoint({'epoch': epoch,
							  'state_dict': model.state_dict(),
							  'optimizer_state': optimizer.state_dict(),
							  'accuracy': np.mean(accuracy),
							  'loss': avg_triplet_loss
							  })
		save_if_best({'epoch': epoch,
					  'state_dict': model.state_dict(),
					  'optimizer_state': optimizer.state_dict(),
					  'accuracy': np.mean(accuracy),
					  'loss': avg_triplet_loss
					  }, np.mean(accuracy))

	create_folder('./log_sweep/roc_plots/')
	create_folder('./log_sweep/roc_plots/valid/')

	# if not os.path.exists('./log/roc_plots/'):
	# 	os.mkdir('./log/roc_plots/')
	# if not os.path.exists('./log/roc_plots/valid/'):
	# 	os.mkdir('./log/roc_plots/valid/')

	plot_roc(fpr, tpr, figure_name='./log_sweep/roc_plots/valid/roc_valid_epoch_{}.png'.format(epoch))
	# plot_roc(fpr, tpr, figure_name='./log/roc_valid_epoch_{}.png'.format(epoch))

	return np.mean(accuracy), np.mean(precision), np.mean(recall), avg_triplet_loss, roc_auc


def main():
	init_wandb_config()

	create_folder('./log_sweep')
	create_folder('./log_sweep/patchwise_logs')

	# if not os.path.isdir('./log'):
	# 	os.mkdir('./log') 

	init_log_just_created("./log_sweep/patchwise_logs/valid_{}.csv".format(args.patch))
	init_log_just_created("./log_sweep/patchwise_logs/train_{}.csv".format(args.patch))

	init_final("./log_sweep/train_final.csv")
	init_final("./log_sweep/valid_final.csv")

	validd = pd.read_csv('log_sweep/patchwise_logs/valid_{}.csv'.format(args.patch))
	max_acc = validd['acc'].max()

	model_architecture = args.model_architecture
	pretrained = args.pretrained
	embedding_dimension = args.embedding_dimension

	optimizer = args.optimizer
	start_epoch = 0
	triplet_loss = TripletLoss(args.margin)#.to(device)
	
	

	# Instantiate model
	model = set_model_architecture(
		model_architecture=model_architecture,
		pretrained=pretrained,
		embedding_dimension=embedding_dimension
	)
	
	# Load model to GPU or multiple GPUs if available
	model, flag_train_multi_gpu = set_model_gpu_mode(model)

	# summary(model, (3, 448, 448))

	# Set optimizer
	optimizer = set_optimizer(
		optimizer=optimizer,
		model=model,
		learning_rate=args.learning_rate
	)
	
	scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
	
	if args.load_best or args.load_last:
		checkpoint = './log_sweep/best_state.pth' if args.load_best else './log_sweep/last_checkpoint.pth'
		print('loading', checkpoint)
#         checkpoint = model.module.load_state_dict(checkpoint['model_state_dict'])
		checkpoint = torch.load(checkpoint)
		modelsaver.current_acc = max_acc
		start_epoch = checkpoint['epoch'] + 1
#         model.module.load_state_dict(checkpoint['state_dict'])
		model.load_state_dict(checkpoint['state_dict'], strict=False)
		print("Stepping scheduler")
		try:
			optimizer.load_state_dict(checkpoint['optimizer_state'])
		except ValueError as e:
			print("Can't load last optimizer")
			print(e)
		if args.continue_step:
			scheduler.step(checkpoint['epoch'])
		print(f"Loaded checkpoint epoch: {checkpoint['epoch']}\n"
			  f"Loaded checkpoint accuracy: {checkpoint['accuracy']}\n"
			  f"Loaded checkpoint loss: {checkpoint['loss']}")

	# data_loaders, data_size = get_dataloader(args.train_root_dir, args.valid_root_dir,
	# 											 args.num_train_triplets, args.num_valid_triplets,
	# 											 args.patch, args.batch_size, args.num_workers)

	model = torch.nn.DataParallel(model)
	
	for epoch in range(start_epoch, args.num_epochs + start_epoch):
		print(110 * '-')
		print('\nEpoch [{}/{}]'.format(epoch, args.num_epochs + start_epoch - 1))

		time0 = time.time()
		data_loaders, data_size = get_dataloader(args.train_root_dir, args.valid_root_dir,
												 args.num_train_triplets, args.num_valid_triplets,
												 args.patch, args.batch_size, args.num_workers)
		

		train_acc, train_precision, train_recall, train_loss, train_roc_auc = train(model, optimizer, triplet_loss, scheduler, epoch, data_loaders, data_size, phase='train')
		valid_acc, valid_precision, valid_recall, valid_loss, valid_roc_auc = valid(model, optimizer, triplet_loss, scheduler, epoch, data_loaders, data_size, phase='valid')
		print(f'Execution time                 = {time.time() - time0}')
	print(110 * '-')
	write_csv(f'log_sweep/train_final.csv', [args.patch, train_acc, train_precision, train_recall, train_loss, train_roc_auc])
	write_csv(f'log_sweep/valid_final.csv', [args.patch, valid_acc, valid_precision, valid_recall, valid_loss, valid_roc_auc])

if __name__ == '__main__':
	main()