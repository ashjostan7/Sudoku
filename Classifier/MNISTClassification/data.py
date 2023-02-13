"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os
from pathlib import Path
import shutil

import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


def check_data(data_path):
    '''
    Checks the data path to ensure all MNIST Dataset files exist.

    Args:
    data_path [str]: path to dataset.

    Returns:
    status [boolean]:   
                    True if all files are present. 
                    False if any/all files are missing. 
                    
    '''
    filenames = ['train-labels-idx1-ubyte', 'train-images-idx3-ubyte', 't10k-labels-idx1-ubyte', 't10k-images-idx3-ubyte']
    # for directory, dirnames, filenames in os.walk(data_path):

    return status

def download_data(data_path):

    data_path = Path(data_path)

    if data_path.exists():
        #data_status = check_data(data_path)
        data_status = True #Place holder for now. 
        print(f'[INFO] Clearing existing data path.')
        shutil.rmtree(data_path)
        download_status = True
    else:
        data_path.mkdir(parents=True, exist_ok=False)
        
        # download_status = False
    print(f"Donloading data to {data_path}")
    #Setting up the train data:
    train_data = datasets.MNIST(root= data_path,
                                train = True,
                                download= download_status,
                                transform=ToTensor(),
                                target_transform=None)

    #Setting up train data:
    test_data = datasets.MNIST(root = data_path,
                            train = False,
                                download= download_status,
                                transform=ToTensor(),
                                target_transform=None)

    print(f"Train: {len(train_data.data)} data points & {len(train_data.targets)} labels")
    print(f"Test: {len(test_data.data)} data points & {len(test_data.targets)} labels")
    print(f"Classes: {train_data.classes}")

    return train_data, test_data


def create_dataloader(
    train_data: torchvision.datasets, 
    test_data: torchvision.datasets, 
    batch_size: int, 
    num_workers: int):


    """Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        transform: torchvision transforms to perform on training and testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
        Example usage:
        train_dataloader, test_dataloader, class_names = 
            = create_dataloaders(train_dir=path/to/train_dir,
                                test_dir=path/to/test_dir,
                                transform=some_transform,
                                batch_size=32,
                                num_workers=4)
    """
    # Get class names
    class_names = train_data.classes

    # Turn images into data loaders
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,)
 
    print(f"Train DataLoader:{train_dataloader}")
    print(f"Train: {len(train_dataloader)} batches of batch size {batch_size}")
    print(f"Test DataLoader:{test_dataloader}")
    print(f"Test: {len(test_dataloader)} batches of batch size {batch_size}")


    return train_dataloader, test_dataloader, class_names

if __name__ == "__main__":

    data_path = Path('../data')
    train, test = download_data(data_path)
    create_dataloader(train, test, 16, os.cpu_count())