import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import cv2
import torch
import numpy as np
import pandas as pd
import requests
from PIL import Image
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from natsort import natsorted
from tqdm import tqdm_notebook
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image
import json

def encode_image(img , transform):
    input_tensor = transform(img)
    input_tensor = input_tensor.half()
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)
    output_tensor = pipe.vae.encode(input_tensor)
    return output_tensor.latent_dist.sample()[0]

def encode_image_avator(img , transform):
    transformed_image = transform(Image.fromarray(np.array(img)[:, :, :3]))
    pil_transformed_image = transforms.ToPILImage()(transformed_image)
    return encode_image(pil_transformed_image, transform)

def calculate_mse(tensor1, tensor2):
    return torch.mean((tensor1 - tensor2) ** 2)

def calculate_pearson_similarity(tensor1, tensor2):
    mean_tensor1 = torch.mean(tensor1)
    mean_tensor2 = torch.mean(tensor2)
    centered_tensor1 = tensor1 - mean_tensor1
    centered_tensor2 = tensor2 - mean_tensor2
    dot_product = torch.sum(centered_tensor1 * centered_tensor2)
    norm_tensor1 = torch.norm(centered_tensor1)
    norm_tensor2 = torch.norm(centered_tensor2)
    similarity = dot_product / (norm_tensor1 * norm_tensor2)
    return similarity

def calculate_cosine_similarity(tensor1, tensor2):
    dot_product = torch.sum(tensor1 * tensor2)
    norm_tensor1 = torch.norm(tensor1)
    norm_tensor2 = torch.norm(tensor2)
    similarity = dot_product / (norm_tensor1 * norm_tensor2)
    return similarity


avator_path = './data/avator/PRIEST'
file_list = os.listdir(avator_path)
folder_list = [item for item in file_list if not '.' in item]

belt_bb = [20, 115, 100, 195]
pants_bb = [20, 150, 100, 320]
shoes_bb = [10, 280, 90, 370]
face_bb = [50, 30, 85, 75]
neck_bb = [50, 65, 85, 85]
cap_bb = [50, 10, 85, 30]
coat_bb = [20, 70, 110, 150]
hair_bb = [30, 20, 90, 60]

bb_dict = {}

folder_list = ['shoes', 'belt', 'pants', 'face', 'neck', 'cap', 'coat', 'hair']
avator_path = './data/avator/PRIEST'

        
for part, bb in zip(folder_list, [belt_bb, pants_bb, shoes_bb, face_bb, neck_bb, cap_bb, coat_bb, hair_bb]):
    bb_dict[part] = bb

print(bb_dict)


transform = transforms.Compose([
    transforms.Resize((16, 16)),  # Resize the image to a desired size
    transforms.ToTensor(),       # Convert the image to a tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
])


# In[ ]:


device = "cuda"
class ImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(('.jpg', '.jpeg', '.png')):
            print("New image file created:", event.src_path)
            pil_image = Image.open(event.src_path)
            prompt = "a men wearing game costume"

            target_size = (256, 256)  # Set your desired size
            resized_image = pil_image.resize(target_size)
            
            images = pipe(prompt=prompt,  image=resized_image, num_inference_steps = 150).images[0]
            print("Generate image")
            plt.imshow(images)
                        
            recommed_dict = {}

            for part in bb_dict:
                part_img = images.crop((bb_dict[part][0], bb_dict[part][1], bb_dict[part][2], bb_dict[part][3]))
                image_lv = encode_image(part_img , transform)

                part_latents = []
                part_avators = []
                folder_list = ['shoes', 'belt', 'pants', 'face', 'neck', 'cap', 'coat', 'hair']
                avator_path = './data/avator/PRIEST'

                full_paths = os.path.join(avator_path, part)
                part_imgs = natsorted(os.listdir(full_paths))
                part_imgs = [file for file in part_imgs if file.lower().endswith('.png')]
                for path in tqdm(part_imgs,desc = f'{part}'):
                    
                    part_img_path = os.path.join(f'./data/avator/PRIEST/{part}' , path)
                    avator_part = Image.open(part_img_path)
                    avator_part = cv2.cvtColor(np.array(avator_part), cv2.COLOR_BGR2RGB)
                    part_latents.append(encode_image_avator(avator_part, transform))
                    part_avators.append(avator_part)
                    
                mse_list = [calculate_mse(image_lv, t).item() for t in image_lv]
                part_rec_mse = np.array(mse_list).argmin()
                plt.imshow(part_avators[part_rec_mse])
                recommed_dict[part] = int(part_rec_mse)
            
            json_data = json.dumps(recommed_dict)
            save_name = event.src_path.split('/')[-1]
            with open(f'./{output_dir}/{save_name}_recommendations.json', 'w') as file:
                file.write(json_data)
            print('create json')


# In[ ]:


if __name__ == "__main__":
    path = "input_image"  # 이미지 파일이 생성될 경로
    output_dir = 'result'

    model_path = "./output/pytorch_lora_weights.bin"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False
    )

    # load lora weights
    pipe.unet.load_attn_procs(model_path)
    # set to use GPU for inference
    pipe.to(device)
    print('load model')

    event_handler = ImageHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()


# In[ ]:




