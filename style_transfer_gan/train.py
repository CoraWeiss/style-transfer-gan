import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from models.generator import Generator
from models.discriminator import Discriminator
from data.data_loader import StyleDataset
import os

def train_gan(data_dir, output_dir, num_epochs=100, batch_size=32, lr=0.0001, 
              latent_dim=100, device='cuda' if torch.cuda.is_available() else 'cpu'):
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = StyleDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)
    
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    criterion = nn.BCELoss()
    
    for epoch in range(num_epochs):
        for i, real_images in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Train discriminator
            d_optimizer.zero_grad()
            output_real = discriminator(real_images)
            d_loss_real = criterion(output_real, 
                                  torch.ones(batch_size, 1, 4, 4).to(device))
            
            noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
            fake_images = generator(noise)
            output_fake = discriminator(fake_images.detach())
            d_loss_fake = criterion(output_fake, 
                                  torch.zeros(batch_size, 1, 4, 4).to(device))
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # Train generator
            g_optimizer.zero_grad()
            output_fake = discriminator(fake_images)
            g_loss = criterion(output_fake, 
                             torch.ones(batch_size, 1, 4, 4).to(device))
            
            g_loss.backward()
            g_optimizer.step()
            
            if i % 10 == 0:
                print(f'Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}] '
                      f'd_loss: {d_loss.item():.4f} g_loss: {g_loss.item():.4f}')
                
        if (epoch + 1) % 5 == 0:
            save_image(fake_images[:16], 
                      os.path.join(output_dir, f'fake_images_epoch_{epoch+1}.png'),
                      nrow=4, normalize=True)

if __name__ == "__main__":
    train_gan('data/style_images', 'outputs')
