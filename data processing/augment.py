from torchvision import transforms
import cv2
from PIL import Image
from tqdm import tqdm
import os
import random


os.chdir("/home/aritram21/Aivian/Cropped_Data/images")

folders = os.listdir()

transf = [transforms.RandomRotation(degrees=10),transforms.ColorJitter(brightness=0.5,hue=0.3),transforms.GaussianBlur(kernel_size=9)]
random.seed(0)

for file in tqdm(folders): 
    data_size = len(os.listdir("/home/aritram21/Aivian/Cropped_Data/images/"+file))
    c = data_size
    original_images = list(os.listdir("/home/aritram21/Aivian/Cropped_Data/images/"+file))
    if data_size<150:
        quotient = int(150/data_size)
        rem = int(150%data_size)
    elif data_size>=150:
        continue
    
    for i in range(quotient-1):
        img_names = random.sample(original_images,data_size)
        for img_name in img_names:
            img_path = "/home/aritram21/Aivian/Cropped_Data/images/"+file+"/"+img_name
            img = Image.open(img_path)
            trans = random.choice(transf)
            aug_img = trans(img)
            aug_img.save("/home/aritram21/Aivian/Cropped_Data/images/"+file+"/aug-"+img_name)
            
    img_names = random.sample(original_images,rem)
    for img_name in img_names:
        img_path = "/home/aritram21/Aivian/Cropped_Data/images/"+file+"/"+img_name
        img = Image.open(img_path)
        trans = random.choice(transf)
        aug_img = trans(img)
        aug_img.save("/home/aritram21/Aivian/Cropped_Data/images/"+file+"/augrem-"+img_name)
    
    data_size = len(os.listdir("/home/aritram21/Aivian/Cropped_Data/images/"+file))
    print("Dataset increasing from {} to {} for class: {}".format(c,data_size,int(file)))
