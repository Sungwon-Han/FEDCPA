import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision
from torchvision.datasets import  CIFAR10, SVHN, ImageFolder, DatasetFolder, utils

import os
import os.path
import logging
import random
import torch

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

    
class TriggerHandler(object):

    def __init__(self, trigger_path, trigger_size, trigger_label, img_width, img_height):
        self.trigger_img = Image.open(trigger_path).convert('RGB')
        self.trigger_size = trigger_size
        self.trigger_img = self.trigger_img.resize((trigger_size, trigger_size))        
        self.trigger_label = trigger_label
        self.img_width = img_width
        self.img_height = img_height

    def put_trigger(self, img):
        img.paste(self.trigger_img, (self.img_width - self.trigger_size, self.img_height - self.trigger_size))
        return img
  
    
class CIFAR10_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
    
    
class CIFAR10Poison_truncated(data.Dataset):
    def __init__(self, args, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.width = 32
        self.height =32
        self.channels = 3
        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)
        self.poisoning_rate = args.poisoning_rate if train else 1.0
        

        self.data, self.target = self.__build_truncated_dataset__()
        indices = range(len(self.target))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        
    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img)
        
        if index in self.poi_indices:
            target = self.trigger_handler.trigger_label
            img = self.trigger_handler.put_trigger(img)
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
        
    
class SVHN_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, split="train", transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.__build_truncated_dataset__()
        

        
    def __build_truncated_dataset__(self):
        svhn_dataobj = SVHN(self.root, self.split, self.transform, self.target_transform, self.download)
        data = svhn_dataobj.data
        target = np.array(svhn_dataobj.labels)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        img, target = self.data[index], int(self.target[index])
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return len(self.data)
    
    
class SVHNPoison_truncated(data.Dataset):
    def __init__(self, args, root, dataidxs=None, split="train", transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.__build_truncated_dataset__()
        
        if split == "train":
            train = True
        else:
            train = False
        
        self.width = 32
        self.height =32
        self.channels = 3
        
        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)
        self.poisoning_rate = args.poisoning_rate if train else 1.0
        

        self.data, self.target = self.__build_truncated_dataset__()
        indices = range(len(self.target))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        
    def __build_truncated_dataset__(self):
        svhn_dataobj = SVHN(self.root, self.split, self.transform, self.target_transform, self.download)
        data = svhn_dataobj.data
        target = np.array(svhn_dataobj.labels)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        img, target = self.data[index], int(self.target[index])
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        
        if index in self.poi_indices:
            target = self.trigger_handler.trigger_label
            img = self.trigger_handler.put_trigger(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return len(self.data)



class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)
            
        self.target = self.samples[:,1]

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.target[index]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)
        
class ImageFolderPoison_custom(DatasetFolder):
    def __init__(self, args, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        
        self.width = 64
        self.height = 64
        self.channels = 3
        
        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)
        self.poisoning_rate = args.poisoning_rate if train else 1.0
        

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)
            
        indices = range(len(self.samples))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))


    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        
        if index in self.poi_indices:
            target = self.trigger_handler.trigger_label
            sample = self.trigger_handler.put_trigger(sample)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)