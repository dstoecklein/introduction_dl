import torch
from torch import nn


def comp_accuracy(model, data_loader, device):
    correct = 0
    wrong = 0
    num_examples = 0

    # turn on eval mode if model Inherits from nn.Module
    if isinstance(model, nn.Module):
        #model.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc. You need to turn off them during model evaluation, and .eval() will do it for you. 
        model.eval()

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)

            # single class with highest probability. simply retain indices
            _, predictions = torch.max(logits, dim=1)

            num_examples += labels.size(0)

            correct += (predictions == labels).sum().float()
            wrong += (predictions != labels).sum().float()

        accuracy = correct / num_examples * 100

    return correct, wrong, accuracy
