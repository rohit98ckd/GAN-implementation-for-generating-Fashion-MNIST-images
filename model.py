import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
latent_dim = 100
hidden_dim = 64
image_dim = 28 * 28
batch_size = 128
lr = 0.0002
beta1 = 0.5
epochs = 20

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, hidden_dim * 2, 4, 2, 1),  # Output: (hidden_dim*2, 14, 14)
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1),  # Output: (hidden_dim, 28, 28)
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, 1, 3, 1, 1),  # Output: (1, 28, 28)
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 4, 2, 1),  # Output: (hidden_dim, 14, 14)
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1),  # Output: (hidden_dim*2, 7, 7)
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(hidden_dim * 2 * 7 * 7, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        return self.model(img)

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss and optimizers
adversarial_loss = nn.BCELoss()
g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Load Fashion-MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_data = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Training
def train_gan(generator, discriminator, train_loader, epochs):
    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(train_loader):
            batch_size = imgs.size(0)
            imgs = imgs.to(device)
            
            # Labels
            real_label = torch.ones(batch_size, 1).to(device)
            fake_label = torch.zeros(batch_size, 1).to(device)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            real_output = discriminator(imgs)
            d_real_loss = adversarial_loss(real_output, real_label)
            
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(z)
            fake_output = discriminator(fake_imgs.detach())
            d_fake_loss = adversarial_loss(fake_output, fake_label)
            
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            fake_output = discriminator(fake_imgs)
            g_loss = adversarial_loss(fake_output, real_label)
            g_loss.backward()
            g_optimizer.step()
            
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}] Batch [{i}/{len(train_loader)}] '
                      f'D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}')
    
    torch.save(generator.state_dict(), 'generator_fmnist.pth')
    torch.save(discriminator.state_dict(), 'discriminator_fmnist.pth')

# Train or load pretrained model
try:
    generator.load_state_dict(torch.load('generator_fmnist.pth'))
    discriminator.load_state_dict(torch.load('discriminator_fmnist.pth'))
    print("Loaded pretrained GAN models")
except FileNotFoundError:
    print("Training new GAN...")
    train_gan(generator, discriminator, train_loader, epochs)
    print("GAN trained and saved")

generator.eval()

# Evaluation
class_mse = {name: [] for name in class_names}

def generate_images(generator, num_samples, latent_dim):
    z = torch.randn(num_samples, latent_dim).to(device)
    with torch.no_grad():
        return generator(z)

with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)
        fake_imgs = generate_images(generator, data.size(0), latent_dim)
        
        for i in range(len(data)):
            original = data[i].cpu().numpy().flatten()
            generated = fake_imgs[i].cpu().numpy().flatten()
            mse = mean_squared_error(original, generated)
            class_mse[class_names[target[i]]].append(mse)

avg_class_mse = {name: np.mean(mses) for name, mses in class_mse.items()}

# Visualization
def show_generated_images(generator, test_loader, class_names, latent_dim, n_samples=5):
    samples = {name: None for name in class_names}
    labels_seen = set()
    
    for data, target in test_loader:
        for i in range(len(data)):
            if class_names[target[i]] not in labels_seen:
                samples[class_names[target[i]]] = data[i]
                labels_seen.add(class_names[target[i]])
                if len(labels_seen) == len(class_names):
                    break
        if len(labels_seen) == len(class_names):
            break
    
    fig, axes = plt.subplots(len(class_names), 2, figsize=(8, 20))
    for idx, (name, sample) in enumerate(samples.items()):
        # Original image
        axes[idx, 0].imshow(sample.squeeze().numpy(), cmap='gray')
        axes[idx, 0].set_title(f'Original: {name}')
        axes[idx, 0].axis('off')
        
        # Generated image
        fake_img = generate_images(generator, 1, latent_dim).squeeze().cpu().numpy()
        axes[idx, 1].imshow(fake_img, cmap='gray')
        axes[idx, 1].set_title(f'Generated (MSE: {avg_class_mse[name]:.4f})')
        axes[idx, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('gan_generations.png')
    plt.show()

show_generated_images(generator, test_loader, class_names, latent_dim)

# Results
print("\nClass-wise Average MSE:")
print("{:<15} {:<10}".format('Class', 'Avg MSE'))
print("-"*25)
for name, mse in sorted(avg_class_mse.items(), key=lambda x: x[1]):
    print("{:<15} {:.4f}".format(name, mse))