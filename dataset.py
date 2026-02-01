import os
import re
import glob
import torch
import datetime
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from sklearn import preprocessing
from torchvision.models import resnet101
from torch.utils.data import Dataset, DataLoader

# フォルダ構成
# mushroom_dataset/
#  ├─ Amanita_muscaria/
#  │   ├─ Amanita_muscaria_001.jpg
#  │   └─ Amanita_muscaria_002.jpg
#  ├─ Grifola_frondosa/
#  │   ├─ Grifola_frondosa_001.jpg
#  │   └─ Grifola_frondosa_002.jpg

class DataSet(Dataset):
    def __init__(self, root):

        l = glob.glob(os.path.join(root, "*", "*.jpg"))
        self.images = []
        self.labels = []
        self.le = preprocessing.LabelEncoder()

        for path in l:
            self.images.append(path)
            filename = os.path.basename(path)
            self.labels.append(filename.split('_')[0])

        self.le.fit(self.labels)
        self.labels_id = self.le.transform(self.labels)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels_id[idx]
        return self.transform(image), int(label)
