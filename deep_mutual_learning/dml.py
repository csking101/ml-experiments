import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import logging
from typing import Optional, Dict, Any, Tuple, List
import timm
import numpy as np
from tqdm import tqdm
from collections import defaultdict

'''
DML
    list of mutual learners
    Experiment
        Dataset
        Loss Function
        Hyperparameter Tuning
    training and weight updation
'''


class Experiment:
    def __init__(
            self, 
            dataset, 
            performance_metric, 
            loss_function,batch_size: int = 32,
            train_val_split: float = 0.8,
            num_workers: int = 4,
            seed: int = 42,
            create_dataloader: bool = True,
            **dataloader_kwargs: Dict[str, Any]):
        
        self.dataset = dataset
        self.loss_function = loss_function
        self.performance_metric = performance_metric
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.num_workers = num_workers
        self.seed = seed
        self.dataloader_kwargs = dataloader_kwargs
        
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if create_dataloader:
            self.create_dataloader()
    
    def create_dataloader(self, test_dataset=None):
        """
        Create train, validation, and optionally test dataloaders.
        
        Args:
            test_dataset: Optional separate test dataset
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        total_size = len(self.dataset)
        train_size = int(total_size * self.train_val_split)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = random_split(
            self.dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed)
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            **self.dataloader_kwargs
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            **self.dataloader_kwargs
        )
        
        if test_dataset is not None:
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                **self.dataloader_kwargs
            )
        
        self.logger.info(f"Created dataloaders with {len(self.train_loader)} training batches "
                        f"and {len(self.val_loader)} validation batches")
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_loaders(self) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """
        Get the created dataloaders.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if self.train_loader is None or self.val_loader is None:
            raise RuntimeError("Dataloaders have not been created. Call create_dataloader() first.")
        return self.train_loader, self.val_loader, self.test_loader
        
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset and splits.
        
        Returns:
            Dictionary containing dataset information
        """
        info = {
            'total_samples': len(self.dataset),
            'train_samples': len(self.train_loader.dataset) if self.train_loader else None,
            'val_samples': len(self.val_loader.dataset) if self.val_loader else None,
            'test_samples': len(self.test_loader.dataset) if self.test_loader else None,
            'batch_size': self.batch_size,
            'train_batches': len(self.train_loader) if self.train_loader else None,
            'val_batches': len(self.val_loader) if self.val_loader else None,
            'test_batches': len(self.test_loader) if self.test_loader else None
        }
        return info
    
    def update_dataloader_params(
        self,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Update dataloader parameters and recreate dataloaders.
        
        Args:
            batch_size: New batch size
            num_workers: New number of workers
            **kwargs: Additional DataLoader parameters to update
        """
        if batch_size is not None:
            self.batch_size = batch_size
        if num_workers is not None:
            self.num_workers = num_workers
            
        self.dataloader_kwargs.update(kwargs)
        self.create_dataloader()
        
        self.logger.info("Updated dataloader parameters and recreated dataloaders")
        

class DML:
    def __init__(
        self,
        models: Dict[str, torch.nn.Module],
        dataset,
        loss_function,
        performance_metric,
        optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
        temperature: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **experiment_kwargs
    ):
        """
        Initialize Deep Mutual Learning framework.
        
        Args:
            models: Dictionary of named models
            dataset: Dataset to use for training/validation
            loss_function: Primary loss function for task
            performance_metric: Metric to evaluate model performance
            optimizers: Dictionary of optimizers for each model (optional)
            temperature: Temperature for softening probability distributions
            device: Device to run computations on
            experiment_kwargs: Additional arguments for Experiment class
        """
        self.models = models
        self.device = device
        self.temperature = temperature
        
        # Move models to device
        for model in self.models.values():
            model.to(self.device)
        
        # Create experiment
        self.experiment = Experiment(
            dataset=dataset,
            loss_function=loss_function,
            performance_metric=performance_metric,
            **experiment_kwargs
        )
        
        # Initialize optimizers if not provided
        self.optimizers = optimizers or {
            name: torch.optim.Adam(model.parameters(), lr=0.001)
            for name, model in self.models.items()
        }
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize tracking metrics
        self.metrics = defaultdict(list)
    
    def kl_loss(self, pred1: torch.Tensor, pred2: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence loss between two predictions.
        
        Args:
            pred1: Predictions from first model
            pred2: Predictions from second model
            
        Returns:
            KL divergence loss
        """
        pred1 = F.log_softmax(pred1 / self.temperature, dim=1)
        pred2 = F.softmax(pred2.detach() / self.temperature, dim=1)
        return F.kl_div(pred1, pred2, reduction='batchmean') * (self.temperature ** 2)
    
    def train_step(
        self,
        batch: tuple,
        mutual_weight: float = 1.0
    ) -> Dict[str, float]:
        """
        Perform one training step for all models.
        
        Args:
            batch: Tuple of (inputs, targets)
            mutual_weight: Weight for mutual learning loss
            
        Returns:
            Dictionary of losses for each model
        """
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            model.train()
            predictions[name] = model(inputs)
        
        # Calculate losses and update each model
        step_losses = {}
        for name, model in self.models.items():
            # Task loss
            task_loss = self.experiment.loss_function(predictions[name], targets)
            
            # Mutual learning loss
            mutual_loss = 0
            for other_name, other_pred in predictions.items():
                if other_name != name:
                    mutual_loss += self.kl_loss(predictions[name], other_pred)
            mutual_loss /= (len(self.models) - 1)  # Average mutual loss
            
            # Total loss
            total_loss = task_loss + mutual_weight * mutual_loss
            step_losses[name] = total_loss.item()
            
            # Optimization step
            self.optimizers[name].zero_grad()
            total_loss.backward(retain_graph=True)  # retain_graph needed for mutual learning
            self.optimizers[name].step()
        
        return step_losses
    
    def train_epoch(
        self,
        mutual_weight: float = 1.0
    ) -> Dict[str, float]:
        """
        Train all models for one epoch.
        
        Args:
            mutual_weight: Weight for mutual learning loss
            
        Returns:
            Dictionary of average losses for each model
        """
        epoch_losses = defaultdict(list)
        
        train_loader, _, _ = self.experiment.get_loaders()
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            step_losses = self.train_step(batch, mutual_weight)
            for name, loss in step_losses.items():
                epoch_losses[name].append(loss)
            
            # Update progress bar
            avg_loss = np.mean(list(step_losses.values()))
            progress_bar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})
        
        # Calculate average losses
        avg_losses = {
            name: np.mean(losses) for name, losses in epoch_losses.items()
        }
        return avg_losses
    
    @torch.no_grad()
    def test_model(
        self,
        name: str,
        loader: Optional[torch.utils.data.DataLoader] = None
    ) -> Dict[str, float]:
        """
        Test a specific model.
        
        Args:
            name: Name of the model to test
            loader: Optional specific loader to use (defaults to validation loader)
            
        Returns:
            Dictionary of test metrics
        """
        if name not in self.models:
            raise ValueError(f"Model {name} not found in DML framework")
        
        model = self.models[name]
        model.eval()
        
        # Use validation loader if no specific loader provided
        if loader is None:
            _, loader, _ = self.experiment.get_loaders()
        
        total_loss = 0
        total_metric = 0
        num_samples = 0
        
        progress_bar = tqdm(loader, desc=f"Testing {name}")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate metrics
            loss = self.experiment.loss_function(outputs, targets)
            metric = self.experiment.performance_metric(outputs, targets)
            
            # Update totals
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_metric += metric.item() * batch_size
            num_samples += batch_size
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{total_loss/num_samples:.4f}",
                "metric": f"{total_metric/num_samples:.4f}"
            })
        
        # Calculate averages
        results = {
            "loss": total_loss / num_samples,
            "metric": total_metric / num_samples
        }
        
        return results
    
    def train(
        self,
        num_epochs: int,
        mutual_weight: float = 1.0,
        validate_every: int = 1
    ) -> Dict[str, List[float]]:
        """
        Train all models for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            mutual_weight: Weight for mutual learning loss
            validate_every: Validate models every N epochs
            
        Returns:
            Dictionary of training history
        """
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Training
            train_losses = self.train_epoch(mutual_weight)
            for name, loss in train_losses.items():
                self.metrics[f"{name}_train_loss"].append(loss)
            
            # Validation
            if (epoch + 1) % validate_every == 0:
                for name in self.models.keys():
                    val_results = self.test_model(name)
                    self.metrics[f"{name}_val_loss"].append(val_results["loss"])
                    self.metrics[f"{name}_val_metric"].append(val_results["metric"])
                    
                    self.logger.info(
                        f"{name} - Val Loss: {val_results['loss']:.4f}, "
                        f"Val Metric: {val_results['metric']:.4f}"
                    )
        
        return dict(self.metrics)
    
    def save_models(self, path: str) -> None:
        """
        Save all models and their optimizers.
        
        Args:
            path: Base path to save models
        """
        for name, model in self.models.items():
            state = {
                'model_state': model.state_dict(),
                'optimizer_state': self.optimizers[name].state_dict()
            }
            torch.save(state, f"{path}/{name}_checkpoint.pt")
        
        self.logger.info(f"Saved all models to {path}")
    
    def load_models(self, path: str) -> None:
        """
        Load all models and their optimizers.
        
        Args:
            path: Base path to load models from
        """
        for name, model in self.models.items():
            checkpoint = torch.load(f"{path}/{name}_checkpoint.pt")
            model.load_state_dict(checkpoint['model_state'])
            self.optimizers[name].load_state_dict(checkpoint['optimizer_state'])
            
        self.logger.info(f"Loaded all models from {path}")
    
