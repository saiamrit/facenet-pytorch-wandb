import numpy as np
import torch
# import visdom
import csv
from path import Path
# from pathlib import Path


class ModelSaver():

    def __init__(self):
        self._previous_acc = 0.
        self._current_acc = 0.

    @property
    def previous_acc(self):
        return self._previous_acc

    @property
    def current_acc(self):
        return self._current_acc

    @current_acc.setter
    def current_acc(self, value):
        self._current_acc = value

    @previous_acc.setter
    def previous_acc(self, value):
        self._previous_acc = value

    def __set_accuracy(self, accuracy):
        self.previous_acc, self.current_acc = self.current_acc, accuracy

    def save_if_best(self, accuracy, state):
        if accuracy > self.current_acc:
            self.__set_accuracy(accuracy)
            torch.save(state, 'log_sweep/best_state.pth')


def create_if_not_exist(path):
    path = Path(path)
    if not path.exists():
        path.touch()


def init_log_just_created(path):
    create_if_not_exist(path)
    with open(path, 'r') as f:
        if len(f.readlines()) <= 0:
            init_log_line(path)

def init_log_line(path):
    with open(path, 'w') as f:
        f.write('time,epoch,acc,precision,recall,loss,roc_auc,batch_size,learning_rate\n')

def write_csv(file, newrow):
    with open(file, mode='a') as f:
        f_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        f_writer.writerow(newrow)

def save_last_checkpoint(state):
    torch.save(state, './log_sweep/last_checkpoint.pth')

# writing into the final file

def init_final(path): #
    create_if_not_exist(path)
    with open(path, 'r') as f:
        if len(f.readlines()) <= 0:
            init_final_line(path)

def init_final_line(path): #
    with open(path, 'w') as f:
        f.write('patch,acc,precision,recall,loss,roc_auc\n')

def write_final(file, newrow): #
    with open(file, mode='a') as f:
        f_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        f_writer.writerow(newrow)
