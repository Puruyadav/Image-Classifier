import argparse
import torch
parser = argparse.ArgumentParser(description="prediction Wow,let's do it.")
parser.add_argument('--image',  type=str,default = 'flowers' + '/test' + '/10/' + 'image_07104.jpg')
parser.add_argument('--checkpoint', type=str,default = "my_folder/checkpoint.pth")
parser.add_argument('--top_k', type=int,default = 5)
parser.add_argument('--category_names', type=str,default = "cat_to_name.json")
parser.add_argument('--gpu', type=bool,default = True)
args = parser.parse_args()


import json
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)


from torchvision import datasets, transforms, models

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]) 
from torchvision import datasets, transforms, models

def load_checkpoint(filepath):
    
    # Checkpoint for when using GPU
    checkpoint = torch.load(filepath)
    model = eval("models." + checkpoint['architecture'] + "(pretrained=True)")
    for param in model.parameters():
        param.requires_grad = False
    
    
    model.classifier = checkpoint['classifier'] 
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model



model = load_checkpoint(args.checkpoint)

device = torch.device("cuda" if args.gpu else "cpu")


from PIL import Image
import numpy as np

def process_image(image):
    
    return np.array(test_transforms(Image.open(image)))

image_name = args.image

import torch.nn.functional as F

def predict(image_path, model, topk=5):
    
    loaded_model = load_checkpoint(model).cpu()
    
    img = process_image(image_path)
    
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    
    img_add_dim = img_tensor.unsqueeze_(0)
    
    loaded_model.eval()
    
    with torch.no_grad():
        
        output = loaded_model.forward(img_add_dim)
    
    probs = torch.exp(output)
    
    probs_top = probs.topk(topk)[0]
    
    index_top = probs.topk(topk)[1]
    
    probs_top_list = np.array(probs_top)[0]
    
    index_top_list = np.array(index_top[0])
    
    class_to_idx = loaded_model.class_to_idx
    
    indx_to_class = {x: y for y, x in class_to_idx.items()}
    
    classes_top_list = []
    
    for index in index_top_list:
        
        classes_top_list += [indx_to_class[index]]
    
    return probs_top_list, classes_top_list

model_path = args.checkpoint

image_path = args.image

probs,classes = predict(image_path, model_path, topk=args.top_k)

import pandas as pd
print(pd.DataFrame(list(zip(classes,probs))).rename(columns = {0:"category",1:"probs"}))

