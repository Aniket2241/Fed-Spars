from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder
import random
import numpy as np


def brain_data(
    path_to_data_folder="/content/fedavg/brain",
    num_clients=10,
    batch_size=32,
    dirichlet_alpha=None,  # If None -> IID, Otherwise Non-IID
    seed=42,
):
    """
    Load and partition Brain Tumor dataset for federated learning (IID or Non-IID using Dirichlet).
    :param path_to_data_folder: Path to brain tumor dataset folder
    :param num_clients: Number of clients
    :param batch_size: Batch size for DataLoaders
    :param dirichlet_alpha: If None -> IID, Otherwise Non-IID degree (e.g., 0.1, 0.3, 0.6)
    :param seed: Random seed for reproducibility
    :return: List of train_loaders, List of test_loaders
    """

    # Transformations (can adjust)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Load dataset
    train_dataset = ImageFolder(root=f"{path_to_data_folder}/train", transform=transform)
    test_dataset = ImageFolder(root=f"{path_to_data_folder}/test", transform=transform)

    # Fix random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    # Extract labels for each sample
    train_labels = np.array(train_dataset.targets)
    test_labels = np.array(test_dataset.targets)

    # --------------------
    # IID Partitioning (Equal Splits)
    # --------------------
    if dirichlet_alpha is None:
        # Shuffle and Split into equal parts
        train_indices = np.random.permutation(len(train_dataset))
        test_indices = np.random.permutation(len(test_dataset))

        train_splits = np.array_split(train_indices, num_clients)
        test_splits = np.array_split(test_indices, num_clients)

    # --------------------
    # Non-IID Partitioning (Dirichlet)
    # --------------------
    else:
        # Helper to split indices using Dirichlet distribution
        def dirichlet_split_indices(labels, n_clients, alpha):
            n_classes = len(set(labels))
            label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

            # Initialize empty lists for each client
            client_indices = [[] for _ in range(n_clients)]

            for c in range(n_classes):
                # Get all indices for class 'c'
                class_indices = np.where(labels == c)[0]
                np.random.shuffle(class_indices)

                # Split class indices based on Dirichlet distribution
                splits = (label_distribution[c] * len(class_indices)).astype(int)

                # Adjust due to rounding (distribute leftover samples)
                diff = len(class_indices) - sum(splits)
                for i in range(diff):
                    splits[i % n_clients] += 1

                # Assign indices to each client
                start = 0
                for i in range(n_clients):
                    end = start + splits[i]
                    client_indices[i].extend(class_indices[start:end])
                    start = end

            # Shuffle within each client to mix different classes
            for i in range(n_clients):
                np.random.shuffle(client_indices[i])

            return client_indices

        train_splits = dirichlet_split_indices(train_labels, num_clients, dirichlet_alpha)
        test_splits = dirichlet_split_indices(test_labels, num_clients, dirichlet_alpha)

    # --------------------
    # Create DataLoaders
    # --------------------
    train_loaders = []
    test_loaders = []

    for i in range(num_clients):
        train_subset = Subset(train_dataset, train_splits[i])
        test_subset = Subset(test_dataset, test_splits[i])

        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, pin_memory=True
        )
        test_loader = DataLoader(
            test_subset, batch_size=batch_size, shuffle=True, pin_memory=True
        )

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return train_loaders, test_loaders