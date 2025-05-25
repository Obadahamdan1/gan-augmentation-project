#  LSGAN for Rain Class (Second GAN Variant)
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

# === Settings ===
image_size = 64
batch_size = 64
latent_dim = 100
n_epochs = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 0.0002

# === Data Loader for Rain Images ===
rain_folder = "C:/Users/obada/.cache/kagglehub/datasets/pratik2901/multiclass-weather-dataset/versions/3/Multi-class Weather Dataset/Rain"
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = ImageFolder(root=os.path.dirname(rain_folder), transform=transform)
rain_images = [sample for sample in dataset.samples if "Rain" in sample[0]]
dataset.samples = rain_images
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === LSGAN Generator ===
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

# === LSGAN Discriminator ===
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 1, 1),
            nn.Flatten()
        )

    def forward(self, x):
        return self.model(x)

# === Initialize ===
G = Generator().to(device)
D = Discriminator().to(device)
criterion = nn.MSELoss()
optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# === Training Loop ===
for epoch in range(n_epochs):
    G.train()
    for real_imgs, _ in loader:
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        gen_imgs = G(z)
        g_loss = criterion(D(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = criterion(D(real_imgs), valid)
        fake_loss = criterion(D(gen_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)
        d_loss.backward()
        optimizer_D.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{n_epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

# === Save Samples ===
os.makedirs("LSGAN_Rain", exist_ok=True)
G.eval()
with torch.no_grad():
    for i in range(500):
        z = torch.randn(1, latent_dim, 1, 1).to(device)
        fake_img = G(z).cpu().squeeze(0)
        img = (fake_img + 1) / 2.0
        img = img.clamp(0, 1)
        save_image(img, f"LSGAN_Rain/rain_lsgan_{i+1:04d}.png")

print("\n✅ LSGAN Training complete — 500 images saved to 'LSGAN_Rain/'")
