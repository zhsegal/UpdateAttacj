import os
import argparse
import time
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from Target_model import ShadowModel, train_shadow, update_model
from torch.autograd import Variable
from CBMGan_new import train_gan, model_train
from torchvision.utils import save_image
import cv2

t0 = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("the device is: {}".format(device))

def generate_update_imgs(gan, encoder, shape, delta_set, num_of_samples, noise_dim, update_set_size):
    model_train(False, gan, encoder)
    num_to_gen = num_of_samples // update_set_size
    imgs = torch.zeros(delta_set.shape[0], num_to_gen, update_set_size, *shape).to(device)
    for i, delta in enumerate(delta_set):
        for j in range(num_to_gen):
            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (1, update_set_size, noise_dim)))).to(device)
            if encoder != None:
                mu = encoder(delta)
            else:
                mu = delta.reshape(1,1,-1).repeat(1, update_set_size, 1)
            gen_img = gan(mu, z)
            imgs[i][j] = gen_img

    im_shape = imgs.shape
    return imgs.reshape(im_shape[0], im_shape[1] * im_shape[2], *im_shape[3:])


def cluster(sets, update_set_size, num_of_samples):

    clusters = np.zeros((sets.shape[0], update_set_size, *sets.shape[2:]))

    for i,imgs in enumerate(sets):
        Z = imgs.reshape((num_of_samples, -1))

        Z = Z.cpu().detach().numpy()

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        K = update_set_size
        dist, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        clusters[i][0] = center[0].reshape(*sets.shape[2:])
        clusters[i][1] = center[1].reshape(*sets.shape[2:])

    return torch.Tensor(clusters)



# get delta for each update set ///need to be done in batches of 1000 for memory reasons
def get_delta(data_set, shadow_model, data, start, end):
    print(start)
    probe_set = data['probe']
    update_sets = data['update']

    if data_set =="Mnist":
        update_set = torchvision.datasets.MNIST(root='./data', train=False,
                                             download=False, transform=data['transform'])
        in_channels = 1
        out_channels = [10, 20]
        fc_size = 320
    elif data_set == "Cifar":
        update_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=False, transform=data['transform'])
        in_channels = data['train'].data.shape[3]
        out_channels=[6,16]
        fc_size = 400
    elif data_set == "STL":
        update_set = torchvision.datasets.STL10(root='./data', split='train',
                                             download=False, transform=data['transform'])
        in_channels = data['train'].data.shape[3]
        out_channels=[6,16]
        fc_size = 400
    
    probe_loader = torch.utils.data.DataLoader(probe_set, batch_size=len(probe_set),
                                             shuffle=False)

    probe_x, _ = next(iter(probe_loader))
    probe_x = probe_x.to(device)
    y_probe = shadow_model(probe_x).view(-1)

    # create m deltas from m updated models
    delta_list = torch.zeros(((end - start), y_probe.shape[0])).to(device)

    new_model = type(shadow_model)(data_set, in_channels, out_channels, fc_size).to(device)  # get a new instance of shadow model

    for i, set in enumerate(update_sets[start:end]):

        # set the data and new model as shadow model
        update_set.data, update_set.targets = set[0], set[1]
        new_model.load_state_dict(shadow_model.state_dict())
        new_model = update_model(new_model, update_set)

        # update the model and get delta
        delta_list[i] = new_model(probe_x).view(-1)
        print(i, end=" ")

    delta_list = y_probe.view(1, -1) - delta_list

    #save data
    torch.save(delta_list, data_set + 'saved/delta_set%d-%d.pt'%(start,end))
    return delta_list


def save(imgs, update_set, name=''):

    for j, set in enumerate(imgs):
        for i, img in enumerate(set):
            if len(img.shape) >2:
                img = img.transpose(0,2)
            save_image(img, ("images%d/" + name + "_gen%d.png") % (j, i), nrow=gan_args.update_set_size, normalize=True)
            im = (update_set[j][0][i].type(torch.FloatTensor) -127.5)/127.5
            save_image(im, ("images%d/" + name + "_real%d.png") % (j, i), nrow=gan_args.update_set_size, normalize=True)


def find_closest(gen_imgs, update_set):

    if type(update_set) is list:
        x = [update_set[index][0].type(torch.FloatTensor) for index in range(len(update_set))]

        update_img = (torch.stack(x) - 127.5) / 127.5
    else:
        update_img = update_set
    shape = gen_imgs.shape
    min_dist = torch.zeros(update_img.shape).to(device)

    # calc MSE to each update image
    for i in range(update_img.shape[0]):
        for j in range(update_img.shape[1]):
            sub = gen_imgs[i] - update_img[i][j].view((1, *shape[2:])).to(device)
            dist = torch.norm(sub.view(shape[1], -1), dim=1).pow(2)
            min_dist[i][j] = gen_imgs[i][torch.argmin(dist)]

    return min_dist

# get all the data sets
def get_data_sets(args):

    probe_set_size = args.probe_set_size
    update_imgs_size_in_set = args.update_set_size
    num_imgs_to_train_shadow = args.shadow_train_set_size
    num_of_update_sets = args.update_sets_num

    if gan_args.data_set == 'Mnist':
        data_set = torchvision.datasets.MNIST
        trans = transforms.Normalize((0.1307,), (0.3081,))
    
    elif gan_args.data_set == 'Cifar':
        data_set = torchvision.datasets.CIFAR10
        trans = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    elif gan_args.data_set == 'STL':
        data_set = torchvision.datasets.STL10
        trans = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    else:
        raise NotImplementedError

    transform = transforms.Compose(
        [transforms.ToTensor(), trans])

    general_set = data_set(root='./data', train=True,
                                             download=True, transform=transform)

    train_set = data_set(root='./data', train=True,
                                            download=True, transform=transform)

    test_set = data_set(root='./data', train=False,
                                             download=True, transform=transform)
    
    probe_set = data_set(root='./data', train=True,
                                            download=True, transform=transform)
   
    update_set = data_set(root='./data', train=True,
                                             download=True, transform=transform)

    # spliting data set

    # save imgs to train shadow model
    end = num_imgs_to_train_shadow
    train_set.data = general_set.data[:end]
    train_set.targets = general_set.targets[:end]

    # imgs for probe set
    start = num_imgs_to_train_shadow
    end = num_imgs_to_train_shadow + probe_set_size
    probe_set.data = general_set.data[start:end]
    probe_set.targets = general_set.targets[start:end]

    # imgs for update set
    start = num_imgs_to_train_shadow + probe_set_size
    update_set.data = general_set.data[start:]
    update_set.targets = general_set.targets[start:]

    # create m update sets for m updated model
    update_sets = []
    for i in range(num_of_update_sets):
        indices = np.random.choice(range(update_set.data.shape[0]), update_imgs_size_in_set)
        update_sets.append((update_set.data[indices], [update_set.targets[index] for index in indices]))

    return {'update': update_sets, 'probe': probe_set, 'test': test_set, 'train': train_set, 'transform':transform}


if __name__ == '__main__':
    np.random.seed(1)
    os.makedirs("images", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_set", type=str, default='Cifar', help="mnist or Cifar or STL10 dataset")
    parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=4e-4, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    parser.add_argument("--noise_dim", type=int, default=100, help="noise to generator dim")
    parser.add_argument("--mu_dim", type=int, default=64, help="mu dim- the encoder output")
    # attack data
    parser.add_argument("--update_set_size", type=int, default=10, help="the update set size")
    parser.add_argument("--update_sets_num", type=int, default=10000, help="number of update sets- num of deltas")
    parser.add_argument("--probe_set_size", type=int, default=100, help="probe set for delta")
    parser.add_argument("--shadow_train_set_size", type=int, default=10000, help="number of training images for shadow model")
    gan_args = parser.parse_args()

    print("setting update data ... time: %d" %int(time.time()-t0))
    data = torch.load(gan_args.data_set +'saved/data' + str(gan_args.update_set_size) + '.pt')
    #data = get_data_sets(gan_args)
    #torch.save(data, gan_args.data_set +'saved/data' + str(gan_args.update_set_size) + '.pt')

    print("training shadow model... time: %d" %int(time.time()-t0))
    #shadow_model, accuracy_list = train_shadow(data['train'], gan_args.data_set, data['transform'], num_epochs=50, lr=0.001, batch_size=64)
    #torch.save(shadow_model, gan_args.data_set +'saved/model_shadow' + str(gan_args.update_set_size) + '.pth')
    shadow_model = torch.load(gan_args.data_set +'saved/model_shadow' + str(gan_args.update_set_size) + '.pth', map_location=torch.device(device))

    print("training update models... time: %d" %int(time.time()-t0))
    #delta_set = torch.load(gan_args.data_set +'saved/delta_set' + str(gan_args.update_set_size) + '.pt', map_location=torch.device(device))
    delta_set = get_delta(gan_args.data_set, shadow_model, data, 0, 1000)#0, gan_args.update_sets_num)

    print("start training gan! time: %d" %int(time.time()-t0))
    #gan, encoder = train_gan(gan_args, delta_set, data['update'], gan_args.data_set)
   # gan = torch.load(gan_args.data_set +'saved/gan' + str(gan_args.update_set_size) + '.pth', map_location=torch.device(device))
    #encoder = torch.load(gan_args.data_set +'saved/encoder' + str(gan_args.update_set_size) + '.pth', map_location=torch.device(device))
   # encoder = None
    print("start reconstruct! time: %d" % int(time.time() - t0))
    num_to_test = 5
    #imgs = generate_update_imgs(gan, encoder, data['train'].data.shape[1:], delta_set[:num_to_test], 20000, gan_args.noise_dim, gan_args.update_set_size)
    #torch.save(imgs, gan_args.data_set +'saved/imgs' + str(gan_args.update_set_size) + '.pth')
    #imgs = torch.load(gan_args.data_set +'saved/imgs' + str(gan_args.update_set_size) + '.pth', map_location=torch.device(device))
    #min_dist = find_closest(imgs, data['update'][:num_to_test])
    #save(min_dist, data['update'][:num_to_test], 'closest10')

    print("start clustering! time: %d" % int(time.time() - t0))
    #clustered_img = cluster(imgs, gan_args.update_set_size, 20000)
    #torch.save(clustered_img, gan_args.data_set +'saved/cluster_imgs.pth')
    
    print("start the end! time: %d" % int(time.time() - t0))
    #closest = find_closest(imgs, clustered_img)
    #save(closest, data['update'][:num_to_test], 'final10')


