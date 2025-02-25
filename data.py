from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision.datasets import ImageFolder
import random
import numpy as np
import os
import shutil


def prepare_dataset(path_to_data_folder, train_ratio=0.8, seed=42):
    """
    Automatically splits dataset into train and test folders if not already done.
    This works for datasets where images are stored inside class-named folders.
    """

    train_path = os.path.join(path_to_data_folder, "train")
    test_path = os.path.join(path_to_data_folder, "test")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"Train-test split already exists in {path_to_data_folder}. Skipping split...")
        return  

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    np.random.seed(seed)
    random.seed(seed)

    for class_name in os.listdir(path_to_data_folder):  
        class_path = os.path.join(path_to_data_folder, class_name)
        if not os.path.isdir(class_path) or class_name in ["train", "test"]:
            continue  

     
        os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_path, class_name), exist_ok=True)

       
        images = os.listdir(class_path)
        random.shuffle(images)  

        split_idx = int(len(images) * train_ratio)  
        train_files, test_files = images[:split_idx], images[split_idx:]

        
        for img in train_files:
            shutil.move(os.path.join(class_path, img), os.path.join(train_path, class_name, img))
        for img in test_files:
            shutil.move(os.path.join(class_path, img), os.path.join(test_path, class_name, img))

    print(f"Train-test split completed successfully for {path_to_data_folder}!")


def load_federated_data(
    dataset_name, path_to_data_folder, num_clients=10, batch_size=32, dirichlet_alpha=None, seed=42
):
    """
    General function to load and partition a dataset (Brain or Alzheimer's) for FL.

    :param dataset_name: Name of the dataset ('brain' or 'alzheimer')
    :param path_to_data_folder: Path to dataset folder
    :param num_clients: Number of clients
    :param batch_size: Batch size for DataLoaders
    :param dirichlet_alpha: If None -> IID, Otherwise Non-IID degree (e.g., 0.1, 0.3, 0.6)
    :param seed: Random seed for reproducibility
    :return: List of train_loaders, List of test_loaders
    """


    if dataset_name in ["alzheimer", "brain"]:
        prepare_dataset(path_to_data_folder)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_dataset = ImageFolder(root=os.path.join(path_to_data_folder, "train"), transform=transform)
    test_dataset = ImageFolder(root=os.path.join(path_to_data_folder, "test"), transform=transform)

    np.random.seed(seed)
    random.seed(seed)

    train_labels = np.array(train_dataset.targets)
    test_labels = np.array(test_dataset.targets)

    # Perform IID or Non-IID Split
    if dirichlet_alpha is None:
        train_splits = np.array_split(np.random.permutation(len(train_dataset)), num_clients)
        test_splits = np.array_split(np.random.permutation(len(test_dataset)), num_clients)
    else:
        train_splits = dirichlet_split_indices(train_labels, num_clients, dirichlet_alpha)
        test_splits = dirichlet_split_indices(test_labels, num_clients, dirichlet_alpha)

    return create_dataloaders(train_dataset, test_dataset, train_splits, test_splits, batch_size)


def dirichlet_split_indices(labels, n_clients, alpha):
    """Helper function to partition dataset in a non-IID manner using Dirichlet distribution."""
    n_classes = len(set(labels))
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    client_indices = [[] for _ in range(n_clients)]

    for c in range(n_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        splits = (label_distribution[c] * len(class_indices)).astype(int)
        diff = len(class_indices) - sum(splits)
        for i in range(diff):
            splits[i % n_clients] += 1

        start = 0
        for i in range(n_clients):
            end = start + splits[i]
            client_indices[i].extend(class_indices[start:end])
            start = end

    for i in range(n_clients):
        np.random.shuffle(client_indices[i])

    return client_indices


def create_dataloaders(train_dataset, test_dataset, train_splits, test_splits, batch_size):
    
    train_loaders = []
    test_loaders = []

    for i in range(len(train_splits)):
        train_subset = Subset(train_dataset, train_splits[i])
        test_subset = Subset(test_dataset, test_splits[i])

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=True, pin_memory=True)

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return train_loaders, test_loaders
