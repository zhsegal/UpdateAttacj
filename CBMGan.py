import torch
import os
import argparse
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
from munkres import Munkres, print_matrix
from torch.autograd import Variable
import matplotlib.pyplot as plt

t0 = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def model_train(train, gen, enc=None, dis=None):

    if train:
        gen.train()
        if enc != None:
            enc.train()
        if dis != None:
            dis.train()
    else:
        gen.eval()
        if enc != None:
            enc.eval()
        if dis != None:
            dis.eval()

class Encoder(nn.Module):
    def __init__(self, mu_dim, num_imgs):
        super(Encoder, self).__init__()
        self.mu_dim = mu_dim
        self.num_imgs = num_imgs
        self.fc1 = nn.Linear(1000, 128)
        self.fc2 = nn.Linear(128, mu_dim)

    def forward(self, delta):
        out = torch.dropout(F.leaky_relu(self.fc1(delta), 0.2), 0.4, self.training)
        mu = torch.dropout(F.leaky_relu(self.fc2(out), 0.2), 0.4, self.training).reshape(-1, 1, self.mu_dim)
        mu = mu.repeat(1, self.num_imgs, 1)
        return mu


class PaperGenerator(nn.Module):
    def __init__(self, img_shape):
        super(PaperGenerator, self).__init__()

        self.img_shape = img_shape
        self.fc1 = nn.Linear(164, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 2048)
        self.fc4 = nn.Linear(2048, 784)

        self.bn = nn.BatchNorm1d(2048)


    def forward(self, mu, noise):
        gen_input = torch.cat((mu, noise), dim=1)
        out = F.leaky_relu(self.bn(self.fc1(gen_input)), 0.2)
        out = F.leaky_relu(self.bn(self.fc2(out)), 0.2)
        out = F.leaky_relu(self.bn(self.fc3(out)), 0.2)
        img = torch.tanh(self.fc4(out))
        img = img.view(img.size(0), *self.img_shape)
        return img

class PaperDiscriminator(nn.Module):
    def __init__(self, img_shape, mu_dim):
        super(PaperDiscriminator, self).__init__()

        self.fc1 = nn.Linear(848, 1024)
        self.bn = nn.BatchNorm1d(2)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)


    def forward(self, img, mu):
        # Concatenate label embedding and image to produce input
        d_input = torch.cat((mu, img.view(img.shape[0], 2, -1)), dim=2)
        out = F.leaky_relu(self.bn(self.fc1(d_input)), 0.2)
        out = F.leaky_relu(self.bn(self.fc2(out)), 0.2)
        out = F.leaky_relu(self.bn(self.fc3(out)), 0.2)
        valid = torch.sigmoid(self.fc4(out))
        return valid


# conditional-gan generator
class Generator(nn.Module):
    def __init__(self, img_shape, mu_dim, noise_dim, update_set_size):
        super(Generator, self).__init__()

        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(update_set_size))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(mu_dim + noise_dim, 128),
            *block(128, 2048),
            *block(2048, 2048),
            *block(2048, 2048),
            nn.Linear(2048, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, mu, noise):
        # Concatenate mu embedding and noise to produce inputs
        gen_input = torch.cat((mu, noise), dim=2)
        imgs = self.model(gen_input)
        imgs = imgs.view(-1, mu.shape[1], *self.img_shape)
        return imgs


class Discriminator(nn.Module):
    def __init__(self, img_shape, mu_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(mu_dim + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, mu):
        d_in = torch.cat((img.reshape(img.shape[0],mu.shape[1], -1), mu), dim=2)
        validity = self.model(d_in)
        return validity


# find closest generated img for each update image using MSE metric
def find_closest_img(batch_size, update_size, update_imgs, gen_imgs, shape):
    gen_img = gen_imgs.reshape(batch_size, update_size, *shape)
    update_img = update_imgs.reshape(batch_size, update_size, *shape)
    min_dist = torch.zeros(update_img.shape).to(device)

    # calc MSE to each update image
    for i in range(update_img.shape[0]):
        for j in range(update_img.shape[1]):
            sub = gen_img[i] - update_img[i][j].view((1, *shape))
            dist = torch.norm(sub.view(update_size, -1), dim=1).pow(2)
            min_dist[i][j] = gen_img[i][torch.argmin(dist)]

    return min_dist

# find closest generated img for each update image using MSE metric and mukres algorithm
def munkres_closest_img(batch_size, update_size, update_imgs, gen_imgs, shape):
    gen_img = gen_imgs.reshape(batch_size, 1, update_size, *shape)
    update_img = update_imgs.reshape(batch_size, update_size, 1, *shape)
    min_dist = torch.zeros(update_img.shape[0], update_size,  *shape).to(device)
    m = Munkres()

    # calc MSE to each update image
    sub = gen_img - update_img
    if len(shape) > 2:
        sub = sub.view(batch_size,update_size,update_size,-1,shape[-1])
    gen_img = gen_img.squeeze()
    dist = torch.norm(sub, dim=(3,4)).pow(2).tolist()
    for i, mat in enumerate(dist):
        indexes = m.compute(mat)
        j = list(list(zip(*indexes))[1])
        min_dist[i] = gen_img[i,j]

    return min_dist.view(update_img.shape[0], update_size,  *shape)

# train the gan with the encoder
def train_gan(args, delta_set, update_set, name=''):

    print(args)

    img_shape = update_set[0][0].shape[1:]

    cuda = True if torch.cuda.is_available() else False

    # Initialize encoder, generator and discriminator
    #encoder = Encoder(args.mu_dim, args.update_set_size)
    #generator = Generator(img_shape, args.mu_dim, args.noise_dim, args.update_set_size) #PaperGenerator(img_shape)#
    generator = Generator(img_shape, 1000, args.noise_dim, args.update_set_size) #PaperGenerator(img_shape)#
   # discriminator = Discriminator(img_shape, args.mu_dim)#PaperDiscriminator(img_shape)#
    discriminator = Discriminator(img_shape, 1000)#PaperDiscriminator(img_shape)#

    # Loss functions
    adversarial_loss = nn.BCELoss()
    bm_loss_mse = nn.MSELoss(reduction="sum")
    bm_loss_bce = nn.BCELoss(reduction="sum")

    if cuda:
        #encoder.cuda()
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Optimizers
    #optimizer_E = torch.optim.Adam(encoder.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=args.lr)

#TODO normalize delata for each image
    # ----------
    #  Training
    # ----------
    total_loss_d = 0
    total_loss_g = 0
    batch_size = args.batch_size
    for epoch in range(args.n_epochs):
        for i in range(delta_set.shape[0] // batch_size):
           # model_train(True, generator, encoder, discriminator)
            model_train(True, generator, dis=discriminator)
            indices = np.random.choice(range(delta_set.shape[0]), batch_size)
            deltas = delta_set[indices]

            # scale update images [-1,1]
            update_imgs = ((torch.stack([update_set[index][0].type(torch.FloatTensor) for index in indices])) - 127.5) / 127.5
            update_imgs = update_imgs.to(device)

            # Adversarial ground truths
            valid = torch.FloatTensor(np.random.uniform(0.7, 1.2, (batch_size, args.update_set_size, 1))).to(device)
            fake = torch.FloatTensor(np.random.uniform(0, 0.3, (batch_size, args.update_set_size, 1))).to(device)

            # -----------------
            #  Train Generator and encoder
            # -----------------

            optimizer_G.zero_grad()
           # optimizer_E.zero_grad()

            # Sample noise
            z = torch.FloatTensor(np.random.normal(0, 1, (batch_size, args.update_set_size, args.noise_dim))).to(device)

            # Generate a batch of images
           # mu = encoder(deltas)
            mu = deltas.unsqueeze(1).repeat(1, args.update_set_size, 1)
            gen_imgs = generator(mu, z)

            # Min MSE for each update image
            closest_gen_imgs = find_closest_img(len(indices), args.update_set_size, update_imgs, gen_imgs, img_shape)
           # closest_gen_imgs = munkres_closest_img(len(indices), args.update_set_size, update_imgs, gen_imgs, img_shape)

            # Loss measures generator
            validity = discriminator(gen_imgs, mu)
            g_loss = bm_loss_mse(closest_gen_imgs, update_imgs) + bm_loss_bce(validity, valid)

            g_loss.backward(retain_graph=True)
            optimizer_G.step()
            #optimizer_E.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(update_imgs, mu)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images (not the min ones)
            validity_fake = discriminator(gen_imgs.detach(), mu)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss

            d_loss.backward()
            optimizer_D.step()

            total_loss_d += (d_loss.item() / batch_size)
            total_loss_g += (g_loss.item() / batch_size)

        total_loss_d /= i
        total_loss_g /= i

        torch.save(generator, name +'saved/gan' + str(args.update_set_size) + '.pth')
      #  torch.save(encoder, name +'saved/encoder' + str(args.update_set_size) + '.pth')
        print(
            "Epoch %d/%d D loss: %f G loss: %f time: %d"
             % (epoch, args.n_epochs, total_loss_d, total_loss_g , int(time.time() -t0))
        )

        if name=="Mnist":
            save_image(gen_imgs.view(-1, 1, 28, 28) , name +"images/gen%d.png" %epoch, nrow=args.update_set_size, normalize=True)
            save_image(update_imgs.view(-1, 1, 28, 28) , name +"images/real%d.png" %epoch, nrow=args.update_set_size, normalize=True)
        else:
            save_image(gen_imgs.view(-1, 32, 32, 3).transpose(1,3), name +"images/gen%d.png" %epoch, nrow=args.update_set_size, normalize=True)
            save_image(update_imgs.view(-1, 32, 32, 3).transpose(1,3), name +"images/real%d.png" %epoch, nrow=args.update_set_size, normalize=True)

  #  return generator, encoder
    return generator, None

if __name__ == '__main__':
    os.makedirs("images", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    args = parser.parse_args()

    train_gan(args)

    #torch.save(model, 'saved/model_' + str(args.mal_target) + '.pth')