# ✅ WGAN for Rain Class (Second GAN Variant)
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import shutil

# === Settings ===
image_size = 64
batch_size = 64
latent_dim = 100
n_epochs = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 0.00005

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

# === WGAN Generator ===
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

# === WGAN Discriminator (called Critic) ===
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
optimizer_G = torch.optim.RMSprop(G.parameters(), lr=lr)
optimizer_D = torch.optim.RMSprop(D.parameters(), lr=lr)

# === WGAN Training Loop ===
n_critic = 5
clip_value = 0.01

for epoch in range(n_epochs):
    for i, (real_imgs, _) in enumerate(loader):
        real_imgs = real_imgs.to(device)

        # === Train Discriminator (Critic) ===
        for p in D.parameters():
            p.requires_grad = True

        for _ in range(n_critic):
            z = torch.randn(real_imgs.size(0), latent_dim, 1, 1).to(device)
            fake_imgs = G(z).detach()

            d_loss = -torch.mean(D(real_imgs)) + torch.mean(D(fake_imgs))

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            for p in D.parameters():
                p.data.clamp_(-clip_value, clip_value)

        # === Train Generator ===
        for p in D.parameters():
            p.requires_grad = False

        z = torch.randn(real_imgs.size(0), latent_dim, 1, 1).to(device)
        fake_imgs = G(z)
        g_loss = -torch.mean(D(fake_imgs))

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{n_epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

# === Save Samples ===
os.makedirs("WGAN_Rain", exist_ok=True)
G.eval()
with torch.no_grad():
    for i in range(500):
        z = torch.randn(1, latent_dim, 1, 1).to(device)
        fake_img = G(z).cpu().squeeze(0)
        img = (fake_img + 1) / 2.0
        img = img.clamp(0, 1)
        save_image(img, f"WGAN_Rain/rain_wgan_{i+1:04d}.png")

print("\n✅ WGAN Training complete — 500 images saved to 'WGAN_Rain/'")

# === Merge into Dataset_WGAN ===
source_dir = "Dataset_Original"
target_dir = "Dataset_WGAN"
os.makedirs(target_dir, exist_ok=True)

# Copy all original folders (Cloudy, Shine, Sunrise, Rain)
for class_name in os.listdir(source_dir):
    source_class_path = os.path.join(source_dir, class_name)
    target_class_path = os.path.join(target_dir, class_name)
    os.makedirs(target_class_path, exist_ok=True)

    for file_name in os.listdir(source_class_path):
        full_source = os.path.join(source_class_path, file_name)
        full_target = os.path.join(target_class_path, file_name)
        shutil.copy2(full_source, full_target)

# Copy WGAN fake rain images into Dataset_WGAN/Rain
wgan_images = os.listdir("WGAN_Rain")
for file_name in wgan_images:
    src = os.path.join("WGAN_Rain", file_name)
    dst = os.path.join(target_dir, "Rain", file_name)
    shutil.copy2(src, dst)

print("\n✅ Dataset_WGAN is ready with real + WGAN Rain data")
