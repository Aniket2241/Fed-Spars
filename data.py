from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder
import random
import numpy as np


def brain_data(path_to_data_folder="/content/Federated-Learning-Sparsification/brain dataset", num_clients=10, train_split=0.8):
    """
    Load custom brain tumor dataset and partition it into `num_clients` clients.
    Assumes directory structure:
    brain_dataset/
        train/
            glioma/
            meningioma/
            pituitary_tumor/
        test/
            glioma/
            meningioma/
            pituitary_tumor/
    """

    # Data transformations (resize images as needed)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to a common size (adjust as needed)
        transforms.ToTensor(),
    ])

    # Load data using ImageFolder
    train_dataset = ImageFolder(root=f"{path_to_data_folder}/train", transform=transform)
    test_dataset = ImageFolder(root=f"{path_to_data_folder}/test", transform=transform)

    # Shuffle data indices for splitting into clients
    train_indices = list(range(len(train_dataset)))
    test_indices = list(range(len(test_dataset)))
    random.seed(42)
    random.shuffle(train_indices)
    random.shuffle(test_indices)

    # Split data into clients
    train_splits = np.array_split(train_indices, num_clients)
    test_splits = np.array_split(test_indices, num_clients)

    all_client_trainloaders = []
    all_client_testloaders = []

    for i in range(num_clients):
        train_subset = Subset(train_dataset, train_splits[i])
        test_subset = Subset(test_dataset, test_splits[i])

        train_loader = DataLoader(dataset=train_subset, batch_size=32, shuffle=True, pin_memory=True)
        test_loader = DataLoader(dataset=test_subset, batch_size=32, shuffle=True, pin_memory=True)

        all_client_trainloaders.append(train_loader)
        all_client_testloaders.append(test_loader)

    return all_client_trainloaders, all_client_testloaders
