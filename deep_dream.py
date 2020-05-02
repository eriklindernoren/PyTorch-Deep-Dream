import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
import tqdm
import scipy.ndimage as nd
from utils import deprocess, preprocess, clip


def dream(image, model, iterations, lr):
    """ Updates the image to maximize outputs for n iterations """
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor
    image = Variable(Tensor(image), requires_grad=True)
    for i in range(iterations):
        model.zero_grad()
        out = model(image)
        loss = out.norm()
        loss.backward()
        avg_grad = np.abs(image.grad.data.cpu().numpy()).mean()
        norm_lr = lr / avg_grad
        image.data += norm_lr * image.grad.data
        image.data = clip(image.data)
        image.grad.data.zero_()
    return image.cpu().data.numpy()


def deep_dream(image, model, iterations, lr, octave_scale, num_octaves):
    """ Main deep dream method """
    image = preprocess(image).unsqueeze(0).cpu().data.numpy()

    # Extract image representations for each octave
    octaves = [image]
    for _ in range(num_octaves - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1, 1 / octave_scale, 1 / octave_scale), order=1))
    
    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(tqdm.tqdm(octaves[::-1], desc="Dreaming")):
        if octave > 0:
            # Upsample detail to new octave dimension
            detail = nd.zoom(detail, np.array(octave_base.shape) / np.array(detail.shape), order=1)
        # Add deep dream detail from previous octave to new base
        input_image = octave_base + detail
        # Get new deep dream image
        dreamed_image = dream(input_image, model, iterations, lr)
        # Extract deep dream details
        detail = dreamed_image - octave_base

    return deprocess(dreamed_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, default="images/supermarket.jpg", help="path to input image")
    parser.add_argument("--iterations", default=20, type=int, help="number of gradient ascent steps per octave")
    parser.add_argument("--at_layer", default=27, type=int, help="layer at which we modify image to maximize outputs")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument("--octave_scale", default=1.4, type=float, help="image scale between octaves")
    parser.add_argument("--num_octaves", default=10, type=int, help="number of octaves")
    args = parser.parse_args()

    # Load image
    image = Image.open(args.input_image)

    # Define the model
    network = models.vgg19(pretrained=True)
    layers = list(network.features.children())
    model = nn.Sequential(*layers[: (args.at_layer + 1)])
    if torch.cuda.is_available:
        model = model.cuda()
    print(network)

    # Extract deep dream image
    dreamed_image = deep_dream(
        image,
        model,
        iterations=args.iterations,
        lr=args.lr,
        octave_scale=args.octave_scale,
        num_octaves=args.num_octaves,
    )

    # Save and plot image
    os.makedirs("outputs", exist_ok=True)
    filename = args.input_image.split("/")[-1]
    plt.figure(figsize=(20, 20))
    # needed to fight rounding errors
    print("before clipping: min", np.min(dreamed_image), "max", np.min(dreamed_image))
    np.clip(dreamed_image, a_min=0.0, a_max=1.0, out=dreamed_image)
    print("after clipping: min", np.min(dreamed_image), "max", np.min(dreamed_image))
    plt.imshow(dreamed_image)
    plt.imsave(f"outputs/output_{filename}", dreamed_image)
    plt.show()
    
