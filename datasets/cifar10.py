from torch.utils import data
from torchvision import transforms
import os
import torch
from PIL import Image

class MYCIFAR10(data.Dataset):
    def __init__(self, **kargs):
        text_list = []
        self.imgs = []
        self.labels= []
        self.img_size = 32
        self.pad_size = int(self.img_size / 8)
        
        if kargs['train']:
            if kargs['augmentation']:
                self.transform = transforms.Compose([
                transforms.RandomCrop(self.img_size, padding = self.pad_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            else:
                self.transform = transforms.ToTensor()
        else:
            self.transform = transforms.ToTensor()
        with open(kargs['list_path'], 'r') as f:
            text_list = f.readlines()
        f.close()
        for i in range(len(text_list)):
            self.imgs.append(os.path.join('data',text_list[i].split(',')[0]))
            self.labels.append(int(text_list[i].split(',')[1]))
        
    def __len__(self):

        return len(self.labels)
    
    def __getitem__(self, index):
        img = Image.open(self.imgs[index])
        img = self.transform(img)
        target = torch.LongTensor([self.labels[index]]).squeeze()
        
        return img, target




        
        

