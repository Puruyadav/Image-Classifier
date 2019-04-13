#lets use argparser to take values from command line
import argparse
parser = argparse.ArgumentParser(description="Artifical neural network")
parser.add_argument('--data_directory', type=str,default = "flowers")
parser.add_argument('--arch', type=str,default = "vgg11")
parser.add_argument('--learning_rate', type=float,default = .001)
parser.add_argument('--hidden_units', type=int,default = 4096)
parser.add_argument('--epochs', type=int,default =5)
parser.add_argument('--gpu', type=bool,default = True)
parser.add_argument('--save_dir', type=str,default = "my_folder")
args = parser.parse_args()



import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

train_dir = args.data_directory + '/train'
valid_dir = args.data_directory + '/valid'
test_dir = args.data_directory + '/test'


train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]) 


test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]) 



train_data = datasets.ImageFolder(train_dir , transform=train_transforms)

validation_data = datasets.ImageFolder(valid_dir , transform=validation_transforms)

test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)

validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=50)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=50)

model = eval("models." + args.arch + "(pretrained=True)")

for param in model.parameters():
    
    param.requires_grad = False
             
             
classifier = nn.Sequential(
                            nn.Linear(25088, args.hidden_units),
                            nn.ReLU(),
                            nn.Dropout(0.4),
                            nn.Linear(args.hidden_units, 102),
                            nn.LogSoftmax(dim=1),
                          )

device = torch.device("cuda" if args.gpu else "cpu")

model.classifier = classifier

model.to(device);

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

epochs = args.epochs

steps = 0

running_loss = 0

print_every = 50

for epoch in range(epochs):
    
    for inputs, labels in train_loader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            validation_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validation_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    validation_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"validation loss: {validation_loss/len(validation_loader):.3f}.. "
                  f"validation accuracy: {accuracy/len(validation_loader):.3f}")
            
            running_loss = 0
            
            model.train()
            

accuary = 0

total = 0

model.to('cuda')

with torch.no_grad():
    
    for image,label in test_loader:
        
        image, label = image.to('cuda'), label.to('cuda')

        outputs = model(image)
        
        _, predicted_outcome = torch.max(outputs.data, 1)

        total += label.size(0)

        accuary += (predicted_outcome == label).sum().item()


print(f"Test accuracy of model: {round(100 * accuary / total,5)}%")

import os

os.makedirs(args.save_dir, exist_ok=True)

os.chdir(args.save_dir)

torch.save({
    
            'state_dict':model.state_dict(),
            'class_to_idx':train_data.class_to_idx,
            'num_epochs': epochs,
            'optimizer_state': optimizer.state_dict,
            'classifier': model.classifier,
             "architecture":args.arch
    
              },

            'checkpoint.pth')            






