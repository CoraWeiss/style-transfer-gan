# Style Transfer GAN

A PyTorch implementation of a Generative Adversarial Network for artistic style transfer, featuring real-time visualization and modular architecture.

## Technical Overview

### Core Technologies
- **Framework**: PyTorch 2.0+ with torchvision
- **Architecture**: DCGAN-based with custom modifications
- **Input Format**: 64x64 RGB images
- **Output**: Style-transferred images in 4x4 grid layouts
- **Training Time**: ~2-3 hours on standard GPU

## Detailed Project Structure
```
style-transfer-gan/
├── training_code/
│   ├── gan_training.py    # Core training loop and loss calculations
│   ├── trainer_config.py  # Hyperparameter configurations
│   └── train_utils.py     # Checkpoint and monitoring utilities
├── models/
│   ├── generator.py       # Generator CNN architecture
│   └── discriminator.py   # Discriminator CNN architecture
├── data/
│   ├── data_loader.py     # Custom dataset handling
│   └── style_images/      # Training image directory
└── outputs/               # Generated images and checkpoints
```

## Installation and Setup

### Prerequisites
```bash
# System requirements
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 50GB disk space for dataset and outputs

# Software requirements
- Python 3.7+
- CUDA toolkit (if using GPU)
```

### Installation Steps
```bash
git clone https://github.com/coraweiss/style-transfer-gan
cd style-transfer-gan
pip install -r requirements.txt

# Optional: Create virtual environment first
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

## Model Architecture

### Generator
- Input: 100D random noise vector
- Output: 64x64x3 RGB image
- Architecture:
```python
Sequential(
    ConvTranspose2d(100 -> 512, kernel=4, stride=1),  # Project and reshape
    BatchNorm2d + ReLU,
    ConvTranspose2d(512 -> 256, kernel=4, stride=2),  # 8x8
    BatchNorm2d + ReLU,
    ConvTranspose2d(256 -> 128, kernel=4, stride=2),  # 16x16
    BatchNorm2d + ReLU,
    ConvTranspose2d(128 -> 64, kernel=4, stride=2),   # 32x32
    BatchNorm2d + ReLU,
    ConvTranspose2d(64 -> 3, kernel=4, stride=2),     # 64x64
    Tanh()
)
```

### Discriminator
- Input: 64x64x3 RGB image
- Output: Binary classification (real/fake)
- Architecture:
```python
Sequential(
    Conv2d(3 -> 64, kernel=4, stride=2),     # 32x32
    LeakyReLU(0.2),
    Conv2d(64 -> 128, kernel=4, stride=2),   # 16x16
    BatchNorm2d + LeakyReLU(0.2),
    Conv2d(128 -> 256, kernel=4, stride=2),  # 8x8
    BatchNorm2d + LeakyReLU(0.2),
    Conv2d(256 -> 1, kernel=4, stride=2),    # 4x4
    Sigmoid()
)
```

## Training System

### Configuration Parameters
```python
training_config = {
    # Core parameters
    'num_epochs': 100,      # Total training epochs
    'batch_size': 32,       # Images per batch
    'learning_rate': 0.0001,# Adam optimizer LR
    
    # Model parameters
    'latent_dim': 100,      # Noise vector dimension
    'image_size': 64,       # Output image size
    'channels': 3,          # RGB channels
    
    # Training controls
    'save_frequency': 5,    # Epochs between saves
    'checkpoint_freq': 50,  # Epochs between checkpoints
    
    # Optimizer settings
    'beta1': 0.5,          # Adam beta1
    'beta2': 0.999,        # Adam beta2
}
```

### Training Process
1. **Data Loading**
   ```python
   dataset = StyleDataset('data/style_images')
   dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
   ```

2. **Training Loop**
   ```python
   # For each epoch:
   - Generate fake images from noise
   - Train discriminator on real/fake images
   - Train generator to fool discriminator
   - Save samples every 5 epochs
   - Log losses every 10 batches
   ```

3. **Loss Functions**
   - Discriminator: Binary Cross Entropy
   - Generator: Binary Cross Entropy with inverted labels

## Visualization System

### Image Generation
- **Format**: PNG files in `outputs/` directory
- **Layout**: 4x4 grid (16 samples)
- **Naming**: `fake_images_epoch_X.png`
- **Frequency**: Every 5 epochs
- **Resolution**: 64x64 pixels per image

### Real-time Monitoring
```python
# Console output format:
Epoch [X/100] Batch [Y/Z] d_loss: 0.XXXX g_loss: 0.XXXX

# Progress indicators:
- Current epoch/batch
- Discriminator loss
- Generator loss
```

### Web Interface
Built with React and Express, featuring:
- Epoch selection (5-100)
- Image grid display
- Training progress visualization
- Real-time loss graphs

### Visualization Code
```python
# Image saving parameters
save_image(
    fake_images[:16],      # Select 16 samples
    os.path.join(output_dir, f'fake_images_epoch_{epoch+1}.png'),
    nrow=4,                # 4x4 grid
    normalize=True         # Normalize pixel values
)
```

## Performance Metrics

### Training Speed
- ~2-3 minutes per epoch on NVIDIA RTX 2080
- ~200ms per batch
- Total training time: 2-3 hours

### Memory Usage
- GPU Memory: ~2GB
- RAM Usage: ~4GB
- Disk Space: ~1GB for 100 epochs

### Loss Convergence
- Discriminator: 0.4-0.6 range
- Generator: 1.2-1.8 range
- Stable training after ~50 epochs

## Acknowledgments and References
- Architecture based on [DCGAN paper](https://arxiv.org/abs/1511.06434) (Radford et al.)
- Training stabilization techniques from [Improved GAN Training](https://arxiv.org/abs/1606.03498)
- Visualization inspired by [Progressive Growing of GANs](https://arxiv.org/abs/1710.10196)
