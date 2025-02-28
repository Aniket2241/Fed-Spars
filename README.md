## FL-TKS
 
 FL-TKS is a federated learning framework designed to improve communication efficiency by sparsifying client-uploaded model updates, thereby significantly reducing 
communication costs in federated learning settings.

## Sparsification Support
FL-TKS supports the following sparsification strategies for communication-efficient model updates:

Top-k Sparsification: Selects the top k% of model parameters with the largest absolute changes after local training.
Random Sparsification: Randomly selects k% of the model parameters for upload.
Threshold-based Sparsification: Selects all model parameters whose absolute change exceeds a given threshold.

## Datasets
Three medical imaging datasets ([data.py](data.py)) were used for the experiments, each representing realistic non-IID and imbalanced data distributions
 using Dirichlet partitioning:

* Augmented Alzheimer MRI 
* Brain tumor 
* Augmented IQ-OTH/NCCD Lung Cancer Dataset


All experiments were implemented in PyTorch using Flower's virtual client engine ([https://github.com/adap/flower](https://github.com/adap/flower)).

## Experimental Setup
The following settings were used for the experiments:

* Number of clients: 3
* Local epochs per round: 5
* Learning rate: 0.01
* Optimizer: SGD
* Data heterogeneity: Non-IID with Dirichlet alpha = 0.3
* Client participation: 1.0(100%)
* Number of communication rounds: 200
* Model architecture: Lightweight CNN (CNN500k)

## Sparsification Levels
The following sparsification levels (k) were tested in the Top-k sparsification experiments:
* 0.1
* 0.2
* 0.3
* 0.4

## Authors
- **[Aniket Bhardwaj](https://github.com/Aniket2241)** (First Author)
- **[Dr.Gousia Habib](https://github.com/gousiya26-I)** (Second Author)
- **[Ritvik Sharma](https://github.com/Ritvik0025)** (Third Author)


**Baseline:**
Federated Averaging [(https://arxiv.org/abs/1602.05629)](https://arxiv.org/abs/1602.05629) using the same set-up as above.

## Simulation

Example command:  
``python simulation.py --dataset_name="alzheimer"   --approach="topk" --sparsify_by=0.1 --num_rounds=200 --epochs=5 --learning_rate=0.01 --dirichlet_alpha=0.3``

Requires version flwr==1.6.0 (Newer versions might have compatibility issues)

| Parameter          | Description                                                                                                                                |
|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| --dataset_name     | Can be ``brain`` or ``alzheimer`` or ``lung``                                                                                              |              |                                                                     |
| --approach         | Can be ``random`` ``topk``or ``threshold``                                                                                                 |
| --sparsify_by      | Fraction of top parameters to retain (e.g., 0.1, 0.2, 0.3, 0.4)                                                                            |
| --num_rounds       | Number of communication rounds                                                                                                             |                 |
| --epochs           | Number of epochs per client per round                                                                                                      |
| --learning_rate    | Learning rate for the model training                                                                                                       |
| --regularisation   | Regularisation/weight decay parameter for the optimiser                                                                                    |
| --dirichlet_alpha  | Alpha parameter controlling data heterogeneity across clients (e.g., 0.3)                                                                  |         |

