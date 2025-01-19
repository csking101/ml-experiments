import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn.functional import cross_entropy
from torchmetrics import Accuracy
import timm
from dml import DML
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Define data transforms
def get_cifar10_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    return train_transform, test_transform

# Create models using different architectures
def create_models():
    # Using different architectures for diversity
    model1 = timm.create_model('resnet18', pretrained=True, num_classes=10)
    model2 = timm.create_model('efficientnet_b0', pretrained=True, num_classes=10)
    model3 = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=10)
    
    models = {
        'resnet18': model1,
        'efficientnet': model2,
        'mobilenet': model3
    }
    
    return models

# Define performance metric
def accuracy_metric(outputs, targets):
    preds = torch.argmax(outputs, dim=1)
    return (preds == targets).float().mean()

def plot_training_history(history, save_path=None):
    """
    Plot training metrics over time for all models.
    
    Args:
        history (dict): Training history dictionary from DML
        save_path (str, optional): Path to save the plot
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Training History', fontsize=16)
    
    # Plot loss
    ax = axes[0]
    model_losses = [(item.split('loss'),value) for item, value in history.items() if item.endswith('loss')]
    for model_name,value in model_losses:
        ax.plot(value, label=f'{model_name}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True)
    
    # Plot accuracy
    ax = axes[1]
    model_metrics = [(item.split('loss'),value) for item, value in history.items() if item.endswith('metric')]
    for model_name,value in model_metrics:
        ax.plot(value, label=f'{model_name}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_model_comparison(dml, test_loader, save_path=None):
    """
    Create a comparison plot of model performances on the test set.
    
    Args:
        dml: DML instance
        test_loader: Test data loader
        save_path (str, optional): Path to save the plot
    """
    # Get test results for each model
    results = {}
    for name in dml.models.keys():
        results[name] = dml.test_model(name, loader=test_loader)
    
    # Prepare data for plotting
    models = list(results.keys())
    accuracies = [results[model]['metric'] * 100 for model in models]
    losses = [results[model]['loss'] for model in models]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot accuracy comparison
    sns.barplot(x=models, y=accuracies, ax=ax1, palette='viridis')
    ax1.set_title('Test Accuracy Comparison')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 100)
    
    # Plot loss comparison
    sns.barplot(x=models, y=losses, ax=ax2, palette='viridis')
    ax2.set_title('Test Loss Comparison')
    ax2.set_ylabel('Loss')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_confusion_matrices(dml, test_dataset, batch_size=128, save_path=None):
    """
    Plot confusion matrices for all models.
    
    Args:
        dml: DML instance
        test_dataset: Test dataset (torchvision dataset)
        batch_size (int): Batch size for the test loader
        save_path (str, optional): Path to save the plot
    """
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Create test loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Get predictions for all models
    predictions = {}
    true_labels = []
    
    # Set models to evaluation mode
    for model in dml.models.values():
        model.eval()
    
    with torch.no_grad():
        for name in dml.models.keys():
            model_preds = []
            if not true_labels:  # Only collect true labels once
                for batch in test_loader:
                    inputs, labels = batch
                    inputs = inputs.to(dml.device)
                    outputs = dml.models[name](inputs)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    model_preds.extend(preds)
                    if not true_labels:
                        true_labels.extend(labels.numpy())
            else:
                for batch in test_loader:
                    inputs, _ = batch
                    inputs = inputs.to(dml.device)
                    outputs = dml.models[name](inputs)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    model_preds.extend(preds)
            predictions[name] = np.array(model_preds)
    
    # Set models back to training mode
    for model in dml.models.values():
        model.train()
    
    # Create confusion matrices
    fig, axes = plt.subplots(1, len(dml.models), figsize=(20, 6))
    for idx, (name, preds) in enumerate(predictions.items()):
        cm = pd.crosstab(true_labels, preds, normalize='index')
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'{name} Confusion Matrix')
        if idx == 0:  # Only show labels for the first plot
            axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
        # Add class labels
        axes[idx].set_xticklabels(classes, rotation=45)
        axes[idx].set_yticklabels(classes, rotation=45)
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get transforms
    train_transform, test_transform = get_cifar10_transforms()
    
    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='../data/cifar10',
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='../data/cifar10',
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create models
    models = create_models()
    
    # Create optimizers with different learning rates
    optimizers = {
        'resnet18': torch.optim.Adam(models['resnet18'].parameters(), lr=0.001),
        'efficientnet': torch.optim.Adam(models['efficientnet'].parameters(), lr=0.001),
        'mobilenet': torch.optim.Adam(models['mobilenet'].parameters(), lr=0.001)
    }
    
    # Initialize DML
    dml = DML(
        models=models,
        dataset=train_dataset,
        loss_function=cross_entropy,
        performance_metric=accuracy_metric,
        optimizers=optimizers,
        temperature=2.0,  # Higher temperature for softer probability distribution
        device=device,
        batch_size=128,
        num_workers=4
    )
    
    # Training configuration
    config = {
        'num_epochs': 2,
        'mutual_weight': 0.5,  # Weight for mutual learning loss
        'validate_every': 1    # Validate after every epoch
    }
    
    # Train the models
    try:
        print("Starting training...")
        history = dml.train(**config)
        print(history)
        
        # Visualize training history
        plot_training_history(history, save_path='./plots/training_history.png')
        
        # Compare model performances
        plot_model_comparison(dml, dml.experiment.test_loader, 
                            save_path='./plots/model_comparison.png')
        
        # Plot confusion matrices
        # plot_confusion_matrices(dml, test_dataset, batch_size=128,
        #                       save_path='./plots/confusion_matrices.png')
        
        # Save models
        dml.save_models('./checkpoints')
        
        # Test final performance
        print("\nFinal Test Performance:")
        for name in models.keys():
            results = dml.test_model(name, loader=dml.experiment.test_loader)
            print(f"{name}: Loss = {results['loss']:.4f}, Accuracy = {results['metric']:.4f}")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving models...")
        dml.save_models('./checkpoints')

if __name__ == "__main__":
    main()