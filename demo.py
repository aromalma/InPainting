import argparse
from pipelines import InPaintDDIM

import torch
import numpy as np
import PIL
from draw import draw
from torchvision import transforms


def main(args):
    transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    ])
    # print(args)
    pipe = InPaintDDIM.from_pretrained(args.model)
    size=pipe.unet.config.sample_size
    print(size)

    mask = torch.permute(torch.tensor(draw(args.image,size),dtype=torch.float32),(2,0,1))
    image = PIL.Image.open(args.image).convert("RGB").resize((size,size))  # Ensure the image is in RGB format
    image_tensor = (transform(image)-0.5)*2


    pipe.to("cuda:0")

    image = pipe(ref_image=image_tensor,mask=mask,num_inference_steps=60)#["sample"]

    image.images[0].save("inpainted_image.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Inpainting using DeepDream with Inpainting Module')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--model', type=str, help='ID of the pretrained model')
    args = parser.parse_args()
    main(args)