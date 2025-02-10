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
    # Cihaz ayarı (macOS CPU-only ortamı için GPU olmayabilir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hiperparametreler
    batch_size = 32
    lr = 0.0002
    num_epochs = 100
    latent_dim = 100  # Rastgele gürültü vektörünün boyutu
    img_size = 224     # Görüntü boyutu
    img_channels = 3  # Renkli görüntüler için 3 kanal

    # Veri seti dönüşümleri
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Veri setini yükleyelim
    try:
        # Healthy klasöründeki tüm görüntüleri listele
        image_files = [f for f in os.listdir('/Users/burkayozdemir/Desktop/myprojects/gan_dataaug/7030/healthy') 
                      if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Görüntüleri yükle
        images = []
        for img_file in image_files:
            img_path = os.path.join('/Users/burkayozdemir/Desktop/myprojects/gan_dataaug/7030/healthy', img_file)
            img = Image.open(img_path).convert('RGB')
            if transform:
                img = transform(img)
            images.append(img)
        
        # Tensor listesini tensor'a çevir
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

    # --- Generator Modeli ---
    class Generator(nn.Module):
        def __init__(self, latent_dim, img_channels):
            super(Generator, self).__init__()
            self.init_size = img_size // 4
            self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

            self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
                nn.Tanh()
            )

        def forward(self, z):
            out = self.l1(z)
            out = out.view(out.shape[0], 128, self.init_size, self.init_size)
            img = self.conv_blocks(out)
            return img

    # --- Discriminator Modeli ---
    class Discriminator(nn.Module):
        def __init__(self, img_channels):
            super(Discriminator, self).__init__()

            def discriminator_block(in_filters, out_filters, bn=True):
                block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
                if bn:
                    block.append(nn.BatchNorm2d(out_filters, 0.8))
                return block

            self.model = nn.Sequential(
                *discriminator_block(img_channels, 16, bn=False),
                *discriminator_block(16, 32),
                *discriminator_block(32, 64),
                *discriminator_block(64, 128),
            )

            # Görüntü boyutunu hesapla
            ds_size = img_size // 2**4
            self.adv_layer = nn.Sequential(
                nn.Linear(128 * ds_size ** 2, 1),
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

    # Kayıp fonksiyonu ve optimizasyon ayarları
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Eğitim için yardımcı fonksiyon
    def save_generated_images(epoch, generator, fixed_noise):
        generator.eval()
        with torch.no_grad():
            fake_imgs = generator(fixed_noise)
            # Görüntüleri (-1,1) aralığından (0,1)'e çekiyoruz
            fake_imgs = (fake_imgs + 1) / 2
            grid = torchvision.utils.make_grid(fake_imgs[:16], nrow=4, normalize=False)
            plt.figure(figsize=(10, 10))
            plt.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)))
            plt.axis("off")
            plt.title(f"Epoch {epoch+1}")
            plt.savefig(f'generated_images_epoch_{epoch+1}.png')
            plt.close()
        generator.train()

    # Sabit gürültü vektörü (görselleştirme için)
    fixed_noise = torch.randn(16, latent_dim, device=device)

    print("Eğitim başlıyor...")
    for epoch in range(num_epochs):
        for i, real_imgs in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            
            # Gerçek ve sahte etiketler
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # Gerçek görüntüler
            real_imgs = real_imgs.to(device)

            # Generator Eğitimi
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_dim, device=device)
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # Discriminator Eğitimi
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

        print(f"[Epoch {epoch+1}/{num_epochs}] Loss_D: {d_loss.item():.4f}, Loss_G: {g_loss.item():.4f}")
        
        # Her 10 epoch'ta bir görüntü kaydet
        if (epoch + 1) % 10 == 0:
            save_generated_images(epoch, generator, fixed_noise)

    print("Eğitim tamamlandı!")

    # Sentetik görüntü üretimi
    print("Sentetik görüntüler üretiliyor...")
    generator.eval()
    num_synthetic_images = 1000  # Her sınıf için üretilecek görüntü sayısı

    # Sentetik görüntüleri kaydetmek için klasör oluştur
    os.makedirs("sentetik_veriler_healthy", exist_ok=True)

    with torch.no_grad():
        for i in range(num_synthetic_images):
            z = torch.randn(1, latent_dim, device=device)
            fake_img = generator(z)
            # Görüntüyü (-1,1) aralığından (0,1)'e çek
            fake_img = (fake_img + 1) / 2
            # Görüntüyü kaydet
            torchvision.utils.save_image(fake_img, f"sentetik_veriler_healthy/sentetik_goruntu_{i+1}.png")

    print(f"Toplam {num_synthetic_images} sentetik görüntü üretildi ve kaydedildi.")

if __name__ == '__main__':
    main()