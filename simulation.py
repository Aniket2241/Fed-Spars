import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18
import torchinfo
import os
import pandas as pd
import scipy
from collections import OrderedDict
import random
from datetime import datetime
from collections import defaultdict
import flwr as fl
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Status, GetParametersIns, GetParametersRes, Code
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
import utils
import models
import data
import argparse
os.chdir("./")
class SparsifyClient(fl.client.Client):
    def __init__(self, 
                 cid, 
                 model, 
                 train_loader, 
                 test_loader, 
                 approach,
                 epochs, 
                 sparsify_by,
                 keep_first_last,
                 learning_rate, 
                 regularisation,
                 model_info):
        
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.approach = approach
        self.epochs = epochs
        self.sparsify_by = sparsify_by
        self.keep_first_last = keep_first_last
        self.learning_rate = learning_rate
        self.regularisation = regularisation
        self.model_info = model_info

    def get_parameters(self, ins) -> GetParametersRes:
        model_parameters = [value.cpu().numpy() for value in self.model.state_dict().values()]
        bytes_parameters = utils.values_to_bytes_list(model_parameters)
        return GetParametersRes(status=Status(code=Code.OK, message="Success"),
                                parameters=bytes_parameters)
    
    def fit(self, ins) -> FitRes:
        global_model_parameters = utils.bytes_to_values_list(ins.parameters.tensors)
        global_state_dict = zip(self.model.state_dict().keys(), global_model_parameters)
        global_state_dict = OrderedDict({key: torch.from_numpy(value) for key, value in global_state_dict})
        self.model.load_state_dict(global_state_dict, strict=True)
        utils.train(model=self.model, 
                    train_loader=self.train_loader, 
                    optimiser="SGD",  
                    lr=self.learning_rate, 
                    epochs=self.epochs,
                    weight_decay=self.regularisation)
        flat_updated_parameters = np.concatenate([layer.cpu().numpy().ravel() for layer in self.model.state_dict().values()])
        flat_global_parameters = np.concatenate([layer.ravel() for layer in global_model_parameters])
        flat_delta_parameters = np.subtract(flat_updated_parameters, flat_global_parameters)
        
        if self.approach=="none":
            delta_bytes = utils.values_to_bytes(flat_delta_parameters)
            return FitRes(status=Status(code=Code.OK, message="Success"),
                          parameters=delta_bytes,
                          num_examples=len(self.train_loader),
                          metrics={})
        
        ## sparsify the parameters using one of three approaches: topk, threshold or random
        if self.approach=="topk": 
            if self.keep_first_last:
                middle_flat_delta_parameters = flat_delta_parameters[self.model_info["indices_first_layer"][-1]+1:self.model_info["indices_last_layer"][0]]
                spars_indices = np.argpartition(np.abs(middle_flat_delta_parameters), -self.model_info["num_to_spars"])[-self.model_info["num_to_spars"]:]
                spars_indices += self.model_info["num_first_layer"]
                spars_indices = self.model_info["indices_first_layer"] + list(spars_indices) + self.model_info["indices_last_layer"]
            else:
                spars_indices = np.argpartition(np.abs(flat_delta_parameters), -self.model_info["num_to_spars"])[-self.model_info["num_to_spars"]:]

        elif self.approach=="threshold":
            normalisation_constant = np.linalg.norm(np.abs(flat_delta_parameters), ord=2)
            if self.keep_first_last:
                middle_flat_delta_parameters = flat_delta_parameters[self.model_info["indices_first_layer"][-1]+1:self.model_info["indices_last_layer"][0]]
                spars_indices = np.argwhere(np.abs(middle_flat_delta_parameters) >= (self.sparsify_by * normalisation_constant)).ravel()
                spars_indices += self.model_info["num_first_layer"]
                spars_indices = self.model_info["indices_first_layer"] + list(spars_indices) + self.model_info["indices_last_layer"]
            else:
                spars_indices = np.argwhere(np.abs(flat_delta_parameters) >= (self.sparsify_by * normalisation_constant)).ravel()

        else: # if approach == "random":
            random.seed(str(flat_updated_parameters[int(self.cid)]))
            if self.keep_first_last:
                spars_indices = np.array(random.sample(range(self.model_info["num_model_params"]-self.model_info["num_first_layer"]-self.model_info["num_last_layer"]), self.model_info["num_to_spars"]))
                spars_indices += self.model_info["num_first_layer"]
                spars_indices = self.model_info["indices_first_layer"] + list(spars_indices) + self.model_info["indices_last_layer"]
            else:
                spars_indices = np.array(random.sample(range(self.model_info["num_model_params"]), self.model_info["num_to_spars"]))

        coo_delta_parameters = np.array([(index, flat_delta_parameters[index]) for index in spars_indices], dtype=np.float32)
        coo_delta_bytes = utils.values_to_bytes(coo_delta_parameters)
        return FitRes(status=Status(code=Code.OK, message="Success"),
                      parameters=coo_delta_bytes,
                      num_examples=len(self.train_loader),
                      metrics={})

    def evaluate(self, ins) -> EvaluateRes:
        global_model_parameters = utils.bytes_to_values_list(ins.parameters.tensors)
        global_state_dict = zip(self.model.state_dict().keys(), global_model_parameters)
        global_state_dict = OrderedDict({key: torch.from_numpy(value) for key, value in global_state_dict})
        self.model.load_state_dict(global_state_dict, strict=True)
        test_loss, test_accuracy = utils.test(self.model, self.test_loader)
        return EvaluateRes(status=Status(code=Code.OK, message="Success"),
                           loss=float(test_loss),
                           num_examples=len(self.test_loader),
                           metrics={"accuracy": float(test_accuracy)})
    
class SparsifyStrategy(fl.server.strategy.Strategy):
    def __init__(self, 
                 global_model, 
                 model_parameters,
                 model_flat_parameters,
                 num_clients, 
                 num_eval_clients,
                 empty_deltas_dict, 
                 layer_shapes, 
                 layer_num_params, 
                 cum_num_params,
                 approach):
        super().__init__()
        self.global_model = global_model
        self.model_parameters = model_parameters
        self.model_flat_parameters = model_flat_parameters
        self.num_clients = num_clients
        self.num_eval_clients = num_eval_clients
        self.empty_deltas_dict = empty_deltas_dict
        self.layer_shapes = layer_shapes
        self.layer_num_params = layer_num_params
        self.cum_num_params = cum_num_params
        self.approach = approach
        
    def initialize_parameters(self, client_manager):
        bytes_parameters = utils.values_to_bytes_list(self.model_parameters)
        return bytes_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        random.seed(server_round)
        clients = client_manager.sample(num_clients=self.num_clients, 
                                        min_num_clients=self.num_clients)
        fit_ins = FitIns(parameters, {})
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        
        if self.approach=="none":
            self.model_flat_parameters += np.mean([utils.bytes_to_values(fit_res.parameters.tensors) for _, fit_res in results], axis=0)
        else:
            client_coo_parameters = defaultdict(list)
            for _, fit_res in results:
                client_coo_array = utils.bytes_to_values(fit_res.parameters.tensors)
                for row in client_coo_array:
                    client_coo_parameters[row[0]].append(row[1])
            for index in client_coo_parameters:
                self.model_flat_parameters[int(index)] += np.mean(client_coo_parameters[index])
                
        shaped_model_parameters = [self.model_flat_parameters[self.cum_num_params[i]:self.cum_num_params[i+1]].reshape(self.layer_shapes[i]) for i in range(len(self.layer_num_params))]
        global_model_bytes = utils.values_to_bytes_list(shaped_model_parameters)
        return global_model_bytes, {} 

    def configure_evaluate(self, server_round, parameters, client_manager):
        random.seed(server_round)
        clients = client_manager.sample(num_clients=self.num_eval_clients, 
                                        min_num_clients=self.num_eval_clients)
        
        evaluate_ins = EvaluateIns(parameters, {})
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(self, server_round, results,failures):
        
        weighted_metrics = [(evaluate_res.num_examples,
                             evaluate_res.num_examples * evaluate_res.loss, 
                             evaluate_res.num_examples * evaluate_res.metrics["accuracy"]) for _, evaluate_res in results]
        total_num_examples = sum([value[0] for value in weighted_metrics])
        aggregated_loss = sum([value[1] for value in weighted_metrics]) / total_num_examples
        aggregated_metrics = {"accuracy":sum([value[2] for value in weighted_metrics]) / total_num_examples}

        return aggregated_loss, aggregated_metrics

    def evaluate(self, server_round, parameters):
       
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate federated learning with sparsification")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--femnist_location", default="femnist_data")
    parser.add_argument("--approach", required=True)
    parser.add_argument("--sparsify_by", type=float, required=True)
    parser.add_argument("--num_rounds", type=int, required=True)
    parser.add_argument("--keep_first_last", default=False, action="store_true")
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--learning_rate", default=0.1, type=float)
    parser.add_argument("--regularisation", default=0, type=float)
    parser.add_argument("--dirichlet_alpha", type=float, default=None, help="Dirichlet alpha for non-IID partitioning. Set None for IID.")

    args = parser.parse_args()
    
    
    if args.dataset_name=="brain":
        model=models.create_model("brain", "CNN500k")
        train_loaders, test_loaders = data.load_federated_data(dataset_name="brain",path_to_data_folder="./brain",num_clients=3,dirichlet_alpha=args.dirichlet_alpha)
        frac_clients = 1.0
        frac_eval_clients = 0.3
    elif args.dataset_name=="alzheimer":
        model=models.create_model("alzheimer", "CNN500k")
        train_loaders, test_loaders = data.load_federated_data(dataset_name="alzheimer",path_to_data_folder="./AugmentedAlzheimerDataset",num_clients=3,dirichlet_alpha=args.dirichlet_alpha)
        frac_clients = 1.0
        frac_eval_clients = 0.3
        
 
    num_clients = int(frac_clients * len(train_loaders))
    num_eval_clients = int(frac_eval_clients * len(train_loaders))
    model_flat_parameters = np.concatenate([layer.cpu().numpy().ravel() for layer in model.state_dict().values()])
    model_params = model.state_dict().values()
    layer_shapes = [layer.cpu().numpy().shape for layer in model_params]
    layer_num_params = [len(layer.cpu().numpy().ravel()) for layer in model_params]
    cum_num_params = np.insert(np.cumsum(layer_num_params), 0, 0)
    num_model_params = sum(layer_num_params)
    empty_deltas_dict = {key: np.nan for key in range(num_model_params)}
    
    model_info = {"num_model_params":num_model_params,
                  "num_first_layer":layer_num_params[0]+layer_num_params[1], 
                  "num_last_layer":layer_num_params[-1],
                  "indices_first_layer":list(range(layer_num_params[0] + layer_num_params[1])),
                  "indices_last_layer":list(range(num_model_params - layer_num_params[-1], num_model_params))}
    if args.keep_first_last:
        model_info["num_to_spars"] = int(num_model_params * args.sparsify_by) - model_info["num_first_layer"] - model_info["num_last_layer"]
    else:
        model_info["num_to_spars"] = int(num_model_params * args.sparsify_by)
        
    client_resources = {"num_gpus": 1, "num_cpus": 1}
    
    
    def client_fn(cid):
        train_loader = train_loaders[int(cid)]
        test_loader = test_loaders[int(cid)]
        return SparsifyClient(cid,        
                              model, 
                              train_loader, 
                              test_loader, 
                              args.approach,
                              args.epochs, 
                              args.sparsify_by,
                              args.keep_first_last,
                              args.learning_rate, 
                              args.regularisation,
                              model_info)
    
    # define the custom strategy
    strategy = SparsifyStrategy(global_model=model,
                                model_parameters=model_params,
                                model_flat_parameters=model_flat_parameters,
                                num_clients=num_clients,
                                num_eval_clients=num_eval_clients,
                                empty_deltas_dict=empty_deltas_dict,
                                layer_shapes=layer_shapes,
                                layer_num_params=layer_num_params,
                                cum_num_params=cum_num_params,
                                approach=args.approach)

    
    start = datetime.now()
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,  
        client_resources=client_resources)
    end = datetime.now()
    time_taken = end-start
    
    results = pd.DataFrame({"time_taken": [time_taken],
                            "dataset": [args.dataset_name], 
                            "frac_clients": [frac_clients],
                            "num_clients": [num_clients],
                            "num_rounds": [args.num_rounds],
                            "approach": [args.approach],
                            "epochs": [args.epochs], 
                            "sparsify_by": [args.sparsify_by], 
                            "keep_first_last": [args.keep_first_last],
                            "learning_rate": [args.learning_rate], 
                            "regularisation": [args.regularisation],
                            "losses": [history.losses_distributed],
                            "accs": [history.metrics_distributed["accuracy"]],
                           })
    if os.path.isfile("results/results.csv"): 
        results.to_csv("results/results.csv", mode="a", index=False, header=False)
    else:
        results.to_csv("results/results.csv", mode="a", index=False, header=True)