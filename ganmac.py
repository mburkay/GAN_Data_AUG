import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

def main():
    # Cihaz ayarı
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hiperparametreler
    batch_size = 32
    lr = 0.0002  # Learning rate'i biraz düşürelim
    num_epochs = 200  # Epoch sayısını artıralım
    latent_dim = 128  # Gürültü vektörünün boyutunu artıralım
    img_size = 224
    img_channels = 3

    # Veri seti dönüşümleri - Daha güçlü augmentasyon ekleyelim
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),  # Hafif rotasyon ekleyelim
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Renk ayarlamaları
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        # Healthy klasöründeki tüm görüntüleri listele
        image_files = [f for f in os.listdir('/Users/burkayozdemir/Desktop/myprojects/gan_dataaug/7030/healthy') 
                      if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        images = []
        for img_file in image_files:
            img_path = os.path.join('/Users/burkayozdemir/Desktop/myprojects/gan_dataaug/7030/healthy', img_file)
            img = Image.open(img_path).convert('RGB')
            if transform:
                img = transform(img)
            images.append(img)
        
        images = torch.stack(images)
        
        print(f"Toplam görüntü sayısı: {len(images)}")
        dataloader = DataLoader(
            images,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

    except Exception as e:
        print(f"Veri setini yükleme hatası: {e}")
        return

    # Generator Modeli - Daha derin ve geniş bir yapı
    class Generator(nn.Module):
        def __init__(self, latent_dim, img_channels):
            super(Generator, self).__init__()
            
            self.init_size = img_size // 8
            self.latent_dim = latent_dim
            
            self.l1 = nn.Sequential(
                nn.Linear(latent_dim, 512 * self.init_size ** 2),
                nn.LeakyReLU(0.2)
            )

            self.conv_blocks = nn.ModuleList([
                nn.BatchNorm2d(512),
                
                nn.Upsample(scale_factor=2),
                nn.Conv2d(512, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                
                nn.Upsample(scale_factor=2),
                nn.Conv2d(256, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
            ])
            
            # Düzeltilmiş final katmanı
            self.final_conv = nn.Conv2d(64, img_channels, 3, stride=1, padding=1)
            self.final_tanh = nn.Tanh()
            
        def forward(self, z):
            out = self.l1(z)
            out = out.view(out.shape[0], 512, self.init_size, self.init_size)
            
            for block in self.conv_blocks:
                out = block(out)
            
            out = self.final_conv(out)
            out = self.final_tanh(out)
            return out

    # Discriminator Modeli - Daha derin ve güçlü bir yapı
    class Discriminator(nn.Module):
        def __init__(self, img_channels):
            super(Discriminator, self).__init__()

            def discriminator_block(in_filters, out_filters, bn=True, kernel_size=4):
                block = [
                    nn.Conv2d(in_filters, out_filters, kernel_size, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25)
                ]
                if bn:
                    block.append(nn.BatchNorm2d(out_filters))
                return block

            self.model = nn.Sequential(
                *discriminator_block(img_channels, 32, bn=False),
                *discriminator_block(32, 64),
                *discriminator_block(64, 128),
                *discriminator_block(128, 256),
                *discriminator_block(256, 512),
            )

            ds_size = img_size // 2**5
            self.adv_layer = nn.Sequential(
                nn.Linear(512 * ds_size ** 2, 1024),
                nn.LeakyReLU(0.2),
                nn.Linear(1024, 1),
                nn.Sigmoid()
            )

        def forward(self, img):
            out = self.model(img)
            out = out.view(out.shape[0], -1)
            validity = self.adv_layer(out)
            return validity

    # Modellerin örneklendirilmesi
    generator = Generator(latent_dim, img_channels).to(device)
    discriminator = Discriminator(img_channels).to(device)

    # Kayıp fonksiyonları - Perceptual loss ekleyelim
    adversarial_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()  # L1 loss ekleyelim

    # Optimizasyon ayarları - Beta değerlerini güncelleyelim
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    def save_generated_images(epoch, generator, fixed_noise):
        generator.eval()
        with torch.no_grad():
            fake_imgs = generator(fixed_noise)
            fake_imgs = (fake_imgs + 1) / 2
            grid = torchvision.utils.make_grid(fake_imgs[:16], nrow=4, normalize=False)
            plt.figure(figsize=(10, 10))
            plt.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)))
            plt.axis("off")
            plt.title(f"Epoch {epoch+1}")
            plt.savefig(f'generated_images_epoch_{epoch+1}.png')
            plt.close()
        generator.train()

    fixed_noise = torch.randn(16, latent_dim, device=device)

    print("Eğitim başlıyor...")
    for epoch in range(num_epochs):
        for i, real_imgs in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            real_imgs = real_imgs.to(device)

            # Generator Eğitimi
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_dim, device=device)
            gen_imgs = generator(z)
            
            # Adversarial ve L1 loss birleşimi
            g_loss_adv = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss_l1 = l1_loss(gen_imgs, real_imgs) * 100  # L1 loss ağırlığı
            g_loss = g_loss_adv + g_loss_l1
            
            g_loss.backward()
            optimizer_G.step()

            # Discriminator Eğitimi
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            if i % 100 == 0:
                print(
                    f"[Epoch {epoch+1}/{num_epochs}] [Batch {i}] "
                    f"[D loss: {d_loss.item():.4f}] "
                    f"[G adv: {g_loss_adv.item():.4f}] "
                    f"[G l1: {g_loss_l1.item():.4f}]"
                )
        
        if (epoch + 1) % 10 == 0:
            save_generated_images(epoch, generator, fixed_noise)

    print("Eğitim tamamlandı!")

    print("Sentetik görüntüler üretiliyor...")
    generator.eval()
    num_synthetic_images = 1000

    os.makedirs("sentetik_veriler_healthy2", exist_ok=True)

    with torch.no_grad():
        for i in range(num_synthetic_images):
            z = torch.randn(1, latent_dim, device=device)
            fake_img = generator(z)
            fake_img = (fake_img + 1) / 2
            torchvision.utils.save_image(fake_img, f"sentetik_veriler_healthy2/sentetik_goruntu_{i+1}.png")

    print(f"Toplam {num_synthetic_images} sentetik görüntü üretildi ve kaydedildi.")

if __name__ == '__main__':
    main()