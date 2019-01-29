# Imports here
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
import torchvision.models as models
import numpy as np
import json
from PIL import Image
from torch.autograd import Variable
import argparse
from collections import OrderedDict
from workspace_utils import active_session
import os
import sys
import pandas as pd

def get_input_args():
    print('get input args')
    parser = argparse.ArgumentParser(description = 'model predict module')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='checkpoint file to load the saved model')
    parser.add_argument('--image', type=str, default='flowers', help='path to the image')
    parser.add_argument('--gpu', type=bool, default='True', help='True: gpu, False: cpu')
    parser.add_argument('--top_k', type=int, default=5, help='top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='mapping of category to real name')
    print('input args complete')
    return parser.parse_args()


def load_checkpoint(checkpoint_path):
    print('load checkpoint begin')
    checkpoint = torch.load(checkpoint_path)
    model = getattr(models, checkpoint['arch'])(pretrained=True)  
#    checkpoint= torch.load(checkpoint_path, map_location=lambda storage, loc: storage) # load in CPU mode   
    for param in model.parameters():
        param.requires_grad = False
        model.classifier = checkpoint['classifier']
        model.class_to_idx = checkpoint['class_to_idx']
        model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
#        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
#        print(model)
        print('load checkpoint end')
        return model
    
def process_image(image):
# Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array
    print('process image begin')
    image = Image.open(image)
    w, h = image.size
    if w > h:
        r = float(w)/float(h)
        size = 256*r, 256
    else:
        r = float(h)/float(w)
        size = 256, 256*r
    image.thumbnail(size) 
    image = image.crop((256//2 - 112, 256//2 - 112, 256//2 + 112, 256//2 + 112))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = np.array(image)
    np_img = img_array/255
    npimage = (np_img - mean)/std
    npimage = npimage.transpose(2, 0, 1)
    print('process image end')    
    return npimage

def predict(image_path, model, device, cat_to_name_file, topk):    
# Predict the top K classes from an image file  
    print('predict begin')
    model.to(device)  
    model.eval()
    torch_img = torch.from_numpy(np.expand_dims(process_image(image_path), axis=0)).type(torch.FloatTensor).to(device)
    
    output = torch.exp(model.forward(torch_img))
    probs, classes = output.topk(topk)

    probs = Variable(probs).cpu().numpy()[0]
    probs = [x for x in probs]
        
    classes = Variable(classes).cpu().numpy()[0]
    classes = [c for c in classes]
    idx_to_classes = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_classes[i] for i in classes]
    print('top classes: ', top_classes)
#    top_classes_int = [int(t) for t in top_classes]
    with open(cat_to_name_file, 'r') as f:
        cat_to_name = json.load(f)
    labels = [cat_to_name[l] for l in top_classes]
    print('predict end') 
    return probs, top_classes, labels

def main():
    args = get_input_args()
    is_gpu = args.gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cpu")
    if is_gpu and use_cuda:
        device = torch.device("cuda:0")
        print(f"Device is set to {device}")
    else:
        device = torch.device("cpu")
        print(f"Device is set to {device}")
    
    model = load_checkpoint(args.checkpoint)
    np_image = process_image(args.image)
    topk_prob, topk_class, topk_label= predict(args.image, model, device, args.category_names, args.top_k)

    print('Flower Name: ', topk_label)
    print('Name Probablity: ', topk_prob)
    print('Predicted top K classes : ', topk_class)
    print('Pridiction Output Complete!')
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)


