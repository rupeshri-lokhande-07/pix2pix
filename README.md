# pix2pix
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import zipfile
import urllib.request

# -----------------------
# Configuration
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1
lr = 2e-4
epochs = 2
image_size = 256

# -----------------------
# Download Facades Dataset
# -----------------------
if not os.path.exists("facades"):
    print("Downloading dataset...")
    url = "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.zip"
    urllib.request.urlretrieve(url, "facades.zip")
    with zipfile.ZipFile("facades.zip", 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Dataset downloaded.")

# -----------------------
# Dataset Loader
# -----------------------
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class FacadesDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.files = os.listdir(root)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.files[index])).convert("RGB")
        w, h = img.size
        input_img = img.crop((0, 0, w//2, h))
        target_img = img.crop((w//2, 0, w, h))
        return self.transform(input_img), self.transform(target_img)

dataset = FacadesDataset("facades/train")
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -----------------------
# Generator (UNet)
# -----------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        def down(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2)
            )

        def up(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            )

        self.d1 = down(3, 64)
        self.d2 = down(64, 128)
        self.d3 = down(128, 256)

        self.u1 = up(256, 128)
        self.u2 = up(128, 64)
        self.u3 = nn.ConvTranspose2d(64, 3, 4, 2, 1)

        self.tanh = nn.Tanh()

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)

        u1 = self.u1(d3)
        u2 = self.u2(u1)
        u3 = self.u3(u2)
        return self.tanh(u3)

# -----------------------
# Discriminator (PatchGAN)
# -----------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 1, 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], 1))

G = Generator().to(device)
D = Discriminator().to(device)

criterion_GAN = nn.BCELoss()
criterion_L1 = nn.L1Loss()

opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# -----------------------
# Training
# -----------------------
print("Training Started...")

for epoch in range(epochs):
    for i, (inp, target) in enumerate(tqdm(loader)):
        inp, target = inp.to(device), target.to(device)

        valid = torch.ones(inp.size(0), 1, 30, 30).to(device)
        fake = torch.zeros(inp.size(0), 1, 30, 30).to(device)

        # Train Generator
        opt_G.zero_grad()
        fake_img = G(inp)
        pred_fake = D(inp, fake_img)
        loss_GAN = criterion_GAN(pred_fake, valid)
        loss_L1 = criterion_L1(fake_img, target)
        loss_G = loss_GAN + 100 * loss_L1
        loss_G.backward()
        opt_G.step()

        # Train Discriminator
        opt_D.zero_grad()
        pred_real = D(inp, target)
        loss_real = criterion_GAN(pred_real, valid)

        pred_fake = D(inp, fake_img.detach())
        loss_fake = criterion_GAN(pred_fake, fake)

        loss_D = (loss_real + loss_fake) * 0.5
        loss_D.backward()
        opt_D.step()

        if i % 200 == 0:
            save_image(fake_img.data[:1], f"output_epoch{epoch}_{i}.png", normalize=True)

    print(f"Epoch [{epoch+1}/{epochs}] Done")

print("Training Completed!")
