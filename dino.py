from json import load
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor,Compose, Resize, Normalize
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torch import nn
import torch.nn.functional as F
from PIL import Image

def get_default_device():
    '''
    Function that return the device available

    Return:
    - CUDA or CPU
    '''
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class Dino:
    '''
    Dino
    - Class that implements DINO feature extraction.
    '''
    def __init__(self) -> None:
        
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8', pretrained=True)
        self.activation = {}

    def get_activation(self,name):
        '''
        Method tha retrieve the output of a given layer

        Params
        - name (str): Layer name
        '''
        def hook(self,input,output):
            self.activaion[name] = output.detach()
        return hook

    def get_attention_matrix(self, img,sequential=False,patch = 8):
        '''
        Method that retrive the attention matrix generated by DINO, for a given image

        Params:
        - img (nd_array): 3-Dimensional array representing an image
        - sequential (Bool): Flag that indicates if the model has a MLP head.
        - patch (int): Moldel patch size
        '''
        device = get_default_device()
        self.model.to(device)
        self.model.blocks[-1].attn.qkv.register_forward_hook(self.get_activation('qkv'))

        transform = Compose([Resize(size=(224, 224)), ToTensor()])
        x = transform(img)
        x = x.cuda()
        logits = self.model(x.unsqueeze(0))
        att_mat = self.activation['qkv']
        input_expanded = self.activation['qkv'][0]


        qkv = input_expanded.reshape(785, 3, 12, 768//12)
        q = qkv[:, 0].permute(1, 0, 2)  # (H=12, N=197, D/H=64)
        k = qkv[:, 1].permute(1, 0, 2)  # (H=12, N=197, D/H=64)
        kT = k.permute(0, 2, 1)  # (H=12, D/H=64, N=197)

        attention_matrix = q @ kT

        return attention_matrix, img

    def __call__(self, img, attention_matrix=False):

        '''
        Method that extract the features by DINO for a given image.

        Params
        - img (nd_array): 3-Dimensional array representing an image
        - attention_matriz (Bool): Flag that indicates if the attention matrix should be return
        '''

        if attention_matrix:
            print("Warning: Attention Matrix feature is not implemented yet.")
        device = get_default_device()
        self.model.to(device)

        transform = Compose([Resize(size=(224, 224))])
        x = transform(img)
        
        if device.type == 'cuda':
            x = x.cuda()

        return self.model(x)

    

#Usage

'''dino = Dino()
img = Image.open('../../IMG_0213.JPG')
features = dino.extract_features(img)
print(features.size())'''