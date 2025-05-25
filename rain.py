import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from PIL import Image
import numpy as np
import os

# 1. Settings
image_size = 64
batch_size = 64
latent_dim = 100
epochs = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. Path to Rain folder
rain_folder = "C:\\Users\\obada\\.cache\\kagglehub\\datasets\\pratik2901\\multiclass-weather-dataset\\versions\\3\\Multi-class Weather Dataset\\Rain"

# 3. Transform images
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 4. Load only Rain images
dataset = ImageFolder(root=os.path.dirname(rain_folder), transform=transform)
rain_images = [sample for sample in dataset.samples if "Rain" in sample[0]]
dataset.samples = rain_images
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 5. Define Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 4, 1, 0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# 6. Define Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),      # 3 channels input (RGB)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),        #  force output to [batch, channels, 1, 1]
            nn.Conv2d(128, 1, 1),            # 1x1 conv → [batch, 1, 1, 1]
            nn.Flatten(),                   # → [batch, 1]
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# 7. Init models
G = Generator().to(device)
D = Discriminator().to(device)
loss_fn = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), lr=0.0002)
opt_D = torch.optim.Adam(D.parameters(), lr=0.0002)

# 8. Training loop
for epoch in range(epochs):
    for real_imgs, _ in loader:
        real_imgs = real_imgs.to(device)

        # === Train Discriminator ===
        z = torch.randn(real_imgs.size(0), latent_dim, 1, 1).to(device)
        fake_imgs = G(z)

        D_real = D(real_imgs)
        D_fake = D(fake_imgs.detach())
        real_labels = torch.ones_like(D_real)
        fake_labels = torch.zeros_like(D_fake)

        loss_D = loss_fn(D_real, real_labels) + loss_fn(D_fake, fake_labels)
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # === Train Generator ===
        D_fake = D(fake_imgs)
        loss_G = loss_fn(D_fake, real_labels)
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1} | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")

# 9. Show generated samples
with torch.no_grad():
    z = torch.randn(16, latent_dim, 1, 1).to(device)
    fake_imgs = G(z).cpu()
    grid = make_grid(fake_imgs, nrow=4, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.title("Generated Rain Images")
    plt.show()

# 1. Folder to save images (will be created if it doesn't exist)
save_dir = "Augmented/Rain"
os.makedirs(save_dir, exist_ok=True)

# 2. Switch generator to evaluation mode
G.eval()

# 3. Generate and save images
num_images = 500
with torch.no_grad():
    for i in range(num_images):
        z = torch.randn(1, latent_dim, 1, 1).to(device)
        fake_img = G(z).cpu().squeeze(0)  # shape: [3, 64, 64]

        # Unnormalize from [-1, 1] → [0, 255]
        img = (fake_img + 1) / 2.0
        img = img.clamp(0, 1)
        np_img = img.mul(255).byte().numpy().transpose(1, 2, 0)  # [H, W, C]

        # Convert to PIL and save
        img_pil = Image.fromarray(np_img)
        filename = os.path.join(save_dir, f"rain_fake_{i+1:04d}.png")
        img_pil.save(filename)
        print(f"✅ Saving: {filename}")  # Print each file as confirmation

print(f"\n✅ Done! Saved {num_images} fake rain images in folder: {save_dir}")
