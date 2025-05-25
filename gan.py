import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Hyperparameters
latent_dim = 10
data_dim = 2
epochs = 1000
batch_size = 64
lr = 0.001

# Create toy minority-class-like data (small cluster)
real_data = torch.randn(500, data_dim) * 0.5 + 2  # centered around (2, 2)

# Simple Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, data_dim)
        )

    def forward(self, z):
        return self.model(z)

# Simple Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Initialize models
generator = Generator()
discriminator = Discriminator()

# Loss and optimizers
criterion = nn.BCELoss()
opt_g = torch.optim.Adam(generator.parameters(), lr=lr)
opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    # === Train Discriminator ===
    # Real samples
    real_samples = real_data[torch.randint(0, len(real_data), (batch_size,))]
    real_labels = torch.ones(batch_size, 1)

    # Fake samples
    z = torch.randn(batch_size, latent_dim)
    fake_samples = generator(z)
    fake_labels = torch.zeros(batch_size, 1)

    # Combine and train
    d_input = torch.cat([real_samples, fake_samples])
    d_labels = torch.cat([real_labels, fake_labels])
    d_preds = discriminator(d_input)
    d_loss = criterion(d_preds, d_labels)

    opt_d.zero_grad()
    d_loss.backward()
    opt_d.step()

    # === Train Generator ===
    z = torch.randn(batch_size, latent_dim)
    gen_samples = generator(z)
    gen_labels = torch.ones(batch_size, 1)  # trick discriminator

    g_preds = discriminator(gen_samples)
    g_loss = criterion(g_preds, gen_labels)

    opt_g.zero_grad()
    g_loss.backward()
    opt_g.step()

    # Print loss occasionally
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# === Visualize Generated Samples ===
with torch.no_grad():
    z = torch.randn(1000, latent_dim)
    generated = generator(z)
    plt.scatter(real_data[:, 0], real_data[:, 1], color='blue', alpha=0.3, label="Real")
    plt.scatter(generated[:, 0], generated[:, 1], color='red', alpha=0.3, label="Fake")
    plt.legend()
    plt.title("Vanilla GAN: Real vs. Generated Samples")
    plt.show()
