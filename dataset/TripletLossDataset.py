import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from skimage import io
from torch.utils.data import Dataset
# import glob
import torch
from glob import glob
import torchvision.transforms as transforms

class TripletFaceDataset(Dataset):
    def __init__(self, root_dir, num_triplets, patch, transform=None):
        self.root_dir = root_dir
        self.num_triplets = num_triplets
        self.transform = transform
        self.patch = patch
        self.triplets = self.generate_triplets()
        
    def prepare_face_dictionary(self):
        face_dictionary = dict()
        folders = sorted(glob(self.root_dir + '/*'))
        for fol in folders:
            fol_name = fol.split('/')[-1]
            files = [fil.split('/')[-1] for fil in sorted(glob(fol + '/{}/*.jpg'.format(self.patch)))]
            face_dictionary[fol_name] = files
        return face_dictionary
    
    def generate_triplets(self):
        triplets = []
        # print(self.root_dir)
        # print(sorted(glob(self.root_dir + '/*')))
        identities = [ids.split('/')[-1] for ids in sorted(glob(self.root_dir + '/*'))]
        face_dictionary = self.prepare_face_dictionary()
        num_triplets = 3
        for _ in tqdm(range(self.num_triplets), desc = 'Data Loading'):
            pos_class = np.random.choice(identities)
            neg_class = np.random.choice(identities)
            while len(face_dictionary[pos_class]) < 2:
                pos_class = np.random.choice(identities)
            while pos_class == neg_class:
                neg_class = np.random.choice(identities)

            if len(face_dictionary[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)
            else:
                ianc = np.random.randint(0, len(face_dictionary[pos_class]))
                ipos = np.random.randint(0, len(face_dictionary[pos_class]))
                while ianc == ipos:
                    ipos = np.random.randint(0, len(face_dictionary[pos_class]))
            ineg = np.random.randint(0, len(face_dictionary[neg_class]))
            triplets.append(
                [
                    face_dictionary[pos_class][ianc],
                    face_dictionary[pos_class][ipos],
                    face_dictionary[neg_class][ineg],
                    pos_class,
                    neg_class
                ]
            )
#             print(ianc, ipos, ineg)
#             print(face_dictionary[pos_class][ianc], face_dictionary[pos_class][ipos],
#                   face_dictionary[neg_class][ineg])
        return triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        anc_id, pos_id, neg_id, pos_class, neg_class = self.triplets[idx]
        anc_img_path = os.path.join(self.root_dir, str(pos_class), 'faces', str(anc_id))
        pos_img_path = os.path.join(self.root_dir, str(pos_class), 'faces', str(pos_id))
        neg_img_path = os.path.join(self.root_dir, str(neg_class), 'faces', str(neg_id))

        anc_img = io.imread(anc_img_path)
        pos_img = io.imread(pos_img_path)
        neg_img = io.imread(neg_img_path)
        pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
        neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))

        sample = {'anc_img': anc_img, 'pos_img': pos_img, 'neg_img': neg_img, 'pos_class': pos_class,
                          'neg_class': neg_class}
        
        if self.transform:
            sample['anc_img'] = self.transform(sample['anc_img'])
            sample['pos_img'] = self.transform(sample['pos_img'])
            sample['neg_img'] = self.transform(sample['neg_img'])

        return sample

def get_dataloader(train_root_dir, valid_root_dir,
                   num_train_triplets, num_valid_triplets,
                   patch, batch_size, num_workers):
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]),
        'valid': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])}

    face_dataset = {
        'train': TripletFaceDataset(root_dir=train_root_dir,
                                    num_triplets=num_train_triplets,
                                    patch=patch,
                                    transform=data_transforms['train']),
        'valid': TripletFaceDataset(root_dir=valid_root_dir,
                                    num_triplets=num_valid_triplets,
                                    patch=patch,
                                    transform=data_transforms['valid'])}

    dataloaders = {
        x: torch.utils.data.DataLoader(face_dataset[x], batch_size=batch_size, shuffle=False, num_workers=num_workers)
        for x in ['train', 'valid']}

    data_size = {x: len(face_dataset[x]) for x in ['train', 'valid']}

    return dataloaders, data_size