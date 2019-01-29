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

def get_input_args():
    """ Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object."""
# Creates parse 
    print('get input args')
    parser = argparse.ArgumentParser(description = 'model training module')
    parser.add_argument('--dir', type=str, default='flowers', help='path to images directory')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='hidden layer units')
    parser.add_argument('--epochs', type=int, default=5, help='num of epochs')
    parser.add_argument('--arch', type=str, default='vgg16', help='model architecture')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='save train model to checkpoint file')
    parser.add_argument('--gpu', type=bool, default='True', help='True: gpu, False: cpu')
# returns parsed argument collection
    print('input args complete')
    return parser.parse_args()

def process_data(train_dir, test_dir, valid_dir):
# Define your transforms for the training, validation, and testing sets
# For the training, you'll want to apply transformations such as random scaling, cropping, and flipping.
# input data is resized to 224x224 pixels
# For validation and testing sets, no transformations, but you'll need to resize then crop the images to the appropriate size
# normalize the means and standard deviations of the images    
    print('begin process data transform')
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(), 
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    data_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(), 
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
# Load the datasets with ImageFolder    
    trainset = datasets.ImageFolder(train_dir, transform = train_transforms)
    testset = datasets.ImageFolder(test_dir, transform = data_transforms)
    validset = datasets.ImageFolder(valid_dir, transform = data_transforms)
# Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=32, shuffle=True)
    print('end process data transform')
    return trainloader, testloader, validloader, trainset, testset, validset

def load_pretrained_model(arch):
# Load a pretrained network
    print('load pretrained model')
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        print('Using vgg16')
    elif arch == "densenet":
        model = models.densenet121(pretrained=True)
        print('Using densenet121')
    else:
        print('Please use vgg16 or desnent only, defaulting to vgg16')
        model = models.vgg16(pretrained=True)
    print('load pretrained model complete')    
    return model

def set_classifier(model, hidden_units):
# Build a feed-forward network using the pretrained network    
    print('set classifier')
    if hidden_units == None:
        hidden_units = 512
    input = model.classifier[0].in_features
    classifier = nn.Sequential(OrderedDict([
                                        ('fc1', nn.Linear(input, hidden_units)),
                                        ('dropout1', nn.Dropout(0.2)),
                                        ('relu1', nn.ReLU()),
                                        ('fc2', nn.Linear(hidden_units, 128)),
                                        ('dropout2', nn.Dropout(0.2)),
                                        ('relu2', nn.ReLU()),
                                        ('logits', nn.Linear(128, 102)),
                                        ('output', nn.LogSoftmax(dim=1))
                                        ]))
    model.classifier = classifier
    print('set classifier complete')
    return model

def train_model(epochs, trainloader, device, model, optimizer, criterion):
# Train the model    
    print('Start train model')
    model.to(device)
    print_every = 40
    steps = 0
    train_loss = 0
    train_accuracy = 0
    if type(epochs) == type(None):
        epochs = 5
        print('epochs = 5')
    for e in range(epochs):
        model.classifier.train()
        train_loss = 0
        train_accuracy = 0
        steps = 0
        float_tensor = torch.FloatTensor if device == 'cpu' else torch.cuda.FloatTensor
        for images, labels in iter(trainloader):
            images, labels = images.to(device), labels.to(device)
            steps += 1
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            train_accuracy += equality.type(float_tensor).mean()
            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Train Loss: {:.4f}".format(train_loss/print_every),
                      "Train Accuracy: {:.4f}".format(train_accuracy/print_every))
                train_loss = 0
                train_accuracy = 0
    print('train model complete')            
    return model

def test(model, testloader, criterion, device):
#    print('in test model')
    test_loss = 0
    test_accuracy = 0
    float_tensor = torch.FloatTensor if device == 'cpu' else torch.cuda.FloatTensor
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        test_accuracy += equality.type(float_tensor).mean()
#    print('exiting test model')
    return test_loss, test_accuracy

def test_model(epochs, model, testloader, device, optimizer, criterion):
    print('test model')
#    if type(epochs) == type(None):
    epochs = 2
#        print('epochs = 2')
    model.to(device)
    e = 0
    steps = 0
    running_loss = 0
    print_every = 10 #40
    for e in range(epochs):
        for images, labels in testloader:
            steps += 1
            torch.no_grad()
            images, labels = images.to(device), labels.to(device)
            output = model.forward(images)
            loss = criterion(output, labels)
            running_loss += loss.item()
            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = test(model, testloader, criterion, device)
                    loss.backward()
                    optimizer.step()
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
                model.train()
                running_loss = 0
    print('Complete testing')
    
def validation(model, validloader, device, criterion):
    valid_loss = 0
    valid_accuracy = 0
    correct = 0
    float_tensor = torch.FloatTensor if device == 'cpu' else torch.cuda.FloatTensor
    for images, labels in validloader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        valid_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        correct += equality.type(float_tensor).mean()
        _, predicted = torch.max(output.data, 1)
        valid_accuracy += (predicted == labels).sum().item()
    return valid_loss, valid_accuracy, correct    
    
def valid_model(epochs, model, validloader, device, criterion):
    print('Validate model')
#    if type(epochs) == type(None):
    epochs = 2
#        print('epochs = 2')
    model.to(device)
    e = 0
    steps = 0
    running_loss = 0
    print_every = 10 #40
    for e in range(epochs):
        for images, labels in validloader:
            images, labels = images.to(device), labels.to(device)
            steps += 1
            torch.no_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            running_loss += loss.item()
            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()
                valid_loss = 0
                accuracy = 0
                orig_accuracy=0
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    valid_loss, accuracy, orig_accuracy = validation(model, validloader, device, criterion)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss: {:.4f}.. ".format(valid_loss/len(validloader)),
#                      "Validation Accuracy: {:.2f}".format(accuracy/len(validloader)),
                      "Validation Accuracy: {:.2f}".format(orig_accuracy/len(validloader)))
                model.train()            
                running_loss = 0
    print('Complete validation')   
    
def save_checkpoint(model, trainset, save_dir, arch, optimizer):
    model.class_to_idx = trainset.class_to_idx
    checkpoint = {'arch': arch,
 #                 'input_size': 25088,
 #                 'output_size': 102,
 #                 'hidden_sizes': [4096, 1024],
 #                 'learning_rate': 0.001,
                  'classifier' : model.classifier,
                  'classifier_state_dict': model.classifier.state_dict(),
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict,
#                  'epochs': 2,
                  'class_to_idx': model.class_to_idx
                 }
    torch.save(checkpoint, save_dir)
    print('checkpoint save complete')
    
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

    data_dir = args.dir #'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    trainloader, testloader, validloader, trainset, testset, validset = process_data(train_dir, test_dir, valid_dir)
    model = load_pretrained_model(args.arch)

    for param in model.parameters():
        param.requires_grad = False

    model = set_classifier(model, args.hidden_units)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    trained_model = train_model(args.epochs, trainloader, device, model, optimizer, criterion)
    test_model(args.epochs, trained_model, testloader, device, optimizer, criterion)
    valid_model(args.epochs, trained_model, validloader, device, criterion)
    save_checkpoint(trained_model, trainset, args.save_dir, args.arch, optimizer)
    print('train.py completed')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)    