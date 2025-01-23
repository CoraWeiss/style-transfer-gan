# style-transfer-gan
# Style Transfer GAN

A PyTorch implementation of a Generative Adversarial Network (GAN) for artistic style transfer.

## Features
- Custom GAN architecture optimized for style transfer
- Real-time training visualization
- Progressive image generation across epochs
- Dataset loader for custom style images

## Installation
```bash
git clone https://github.com/coraweiss/style-transfer-gan
cd style-transfer-gan
pip install -r requirements.txt
```

## Usage
1. Add style images to `data/style_images/`
2. Start training:
```bash
cd style_transfer_gan
python train.py
```

Generated images appear in `outputs/` every 5 epochs.

## Model Architecture
- Generator: Transposed convolutional layers with batch normalization
- Discriminator: Convolutional layers with LeakyReLU activation
- Input: 64x64 RGB images
- Latent space: 100-dimensional noise vector

## Training Parameters
- Learning rate: 0.0001
- Batch size: 32
- Epochs: 100
- Optimizer: Adam (β1=0.5, β2=0.999)

## Requirements
- Python 3.7+
- PyTorch 2.0+
- torchvision
- numpy
- Pillow

## Generated Examples
Generated images show progression from noise to learned style across training epochs. See `outputs/` for full results.

## Acknowledgments
Architecture inspired by DCGAN paper (Radford et al.).
