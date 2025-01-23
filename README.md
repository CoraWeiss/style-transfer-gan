# Style Transfer GAN

PyTorch GAN implementation for artistic style transfer.

## Features
- Custom GAN architecture
- Real-time training visualization
- Progressive image generation
- Configurable training parameters
- Modular training code structure

## Installation
```bash
git clone https://github.com/coraweiss/style-transfer-gan
cd style-transfer-gan
pip install -r requirements.txt
```

## Project Structure
```
style-transfer-gan/
├── training_code/
│   ├── gan_training.py    # Core training loop
│   ├── trainer_config.py  # Training parameters
│   └── train_utils.py     # Training utilities
├── models/
│   ├── generator.py       # Generator architecture
│   └── discriminator.py   # Discriminator architecture
└── data/
    └── data_loader.py     # Dataset handling
```

## Usage
1. Add style images to `data/style_images/`
2. Configure parameters in `training_code/trainer_config.py`
3. Start training:
```bash
python train.py
```

## Training Parameters
```python
training_config = {
    'num_epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.0001,
    'latent_dim': 100,
    'image_size': 64,
    'channels': 3,
    'save_frequency': 5
}
```

## Model Architecture
- Generator: Transposed CNN with batch normalization
- Discriminator: CNN with LeakyReLU
- Input: 64x64 RGB images
- Latent space: 100-dimensional

## Requirements
- Python 3.7+
- PyTorch 2.0+
- torchvision
- numpy
- Pillow

## Training Progress
- Generated images saved every 5 epochs in `outputs/`
- Training metrics logged during execution
- Checkpoint saving functionality included

## Acknowledgments
Architecture inspired by DCGAN paper (Radford et al.).
