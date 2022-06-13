
import os
import numpy
import argparse
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from model import *
from utils import *

torch.multiprocessing.set_sharing_strategy('file_system')

def main(args):

    # Args
    batch_size = args.batch_size
    pretrained_model_path = args.pretrained_model_path
    save_path = args.save_path

    # Save Paths
    save_path_embeddings = save_path + 'cifar10_embeddings.pt'
    save_path_labels = save_path + 'cifar10_labels.pt'

    # Load Dataset
    print('Loading Dataset...')
    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Done')

    # Load Model
    print('Loading Model...')
    model = torch.load(pretrained_model_path, map_location='cpu')
    model = model.encoder
    model.to(device)
    print('Done')

    # Process CiFar10 to Create Latent Space Embeddings
    print('Creating Embeddings...')
    model.eval()
    with torch.no_grad():
        dataset_embeddings = []
        data_set_labels = []
        for img, label in tqdm(iter(train_dataloader)):
            img = img.to(device)
            features, _ = model(img)
            features = features[0].squeeze()
            dataset_embeddings.append(features)
            data_set_labels.append(label[0])
        print('Done')

        # Save Embeddings to Disk
        print('Saving Embeddings to Disk...')
        dataset_embeddings = torch.stack(dataset_embeddings)
        data_set_labels = torch.stack(data_set_labels)
        torch.save(dataset_embeddings, save_path_embeddings)
        torch.save(data_set_labels, save_path_labels)
        print('Done')

    print(f'Cifar10 embeddings were encoded with MAE and saved to "cifar10_embeddings" and "cifar10_labels".')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--pretrained_model_path', type=str, default='vit-t-mae.pt')
    parser.add_argument('--save_path', type=str, default='./')

    args = parser.parse_args()

    main(args)