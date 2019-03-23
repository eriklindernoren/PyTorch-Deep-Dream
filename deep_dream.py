import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
import scipy.ndimage as nd
from utils import deprocess, preprocess, clip


def dream(image, model, iterations, lr, guide_image=None):
    image = Variable(torch.cuda.FloatTensor(image), requires_grad=True)
    # Extract guide features if guide image is available
    if guide_image is not None:
        h, w = image.shape[-2:]
        guide_image = preprocess(guide_image.resize((w, h))).unsqueeze(0)
        guide_image = Variable(guide_image.type(torch.cuda.FloatTensor))
        guide_features = model(guide_image).detach().data
    # Update image for n iterations
    for i in range(iterations):
        model.zero_grad()
        out = model(image)
        loss = out.norm() - nn.MSELoss()(out, guide_features) if guide_image is not None else out.norm()
        loss.backward()
        ratio = np.abs(image.grad.data.cpu().numpy()).mean()
        norm_lr = lr / ratio
        image.data += norm_lr * image.grad.data
        image.data = clip(image.data)

    return image.cpu().data.numpy()


def deep_dream(image, model, iterations, lr, octave_scale, num_octaves, guide_image=None):
    image = preprocess(image).unsqueeze(0)

    # Extract octaves for each dimension
    octaves = [image.cpu().data.numpy()]
    for i in range(num_octaves - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1, 1.0 / octave_scale, 1.0 / octave_scale), order=1))

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        # Upsample detail to new dimension
        if octave > 0:
            dh, dw = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1, 1.0 * h / dh, 1.0 * w / dw), order=1)
        # Add deep dream detail from previous octave to new base
        input_oct = octave_base + detail
        # Get new deep dream image
        dreamed_image = dream(input_oct, model, iterations, lr, guide_image)
        # Extract deep dream details
        detail = dreamed_image - octave_base

    return deprocess(dreamed_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="images/supermarket.jpg", help="path to input image")
    parser.add_argument("--guide_path", type=str, default="images/dog.jpg", help="path to guide image")
    parser.add_argument("--iterations", default=15, help="number of gradient ascent steps per octave")
    parser.add_argument("--lr", default=0.01, help="learning rate")
    parser.add_argument("--octave_scale", default=1.4, help="image scale between octaves")
    parser.add_argument("--num_octaves", default=10, help="number of octaves")
    args = parser.parse_args()

    # Load images
    image = Image.open(args.image_path)
    guide_image = Image.open(args.image_path) if args.guide_path else None

    # Define the model
    network = models.vgg19(pretrained=True)
    layers = list(network.features.children())
    model = nn.Sequential(*layers[:28])
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
        guide_image=guide_image,
    )

    # Save and plot image
    os.makedirs("outputs", exist_ok=True)
    filename = args.image_path.split("/")[-1]
    plt.figure(figsize=(20, 20))
    plt.imshow(dreamed_image)
    plt.imsave(f"outputs/output_{filename}.jpg", dreamed_image)
    plt.show()
