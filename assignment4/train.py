import torch
from torch import nn
import time
from metrics import comp_accuracy  # these are our created files


def fit(model, train_loader, epochs, learning_rate, device, loss_func=nn.CrossEntropyLoss(), opt_func=torch.optim.SGD):

    # objective function
    optimizer = opt_func(model.parameters(), learning_rate)

    model = model.to(device)

    print('Training on: {}'.format(device), '\n')

    start = time.time()  # measure time

    for epoch in range(epochs):

        model = model.train()

        for batch_index, (features, labels) in enumerate(train_loader):

            # gpu usage if possible
            features = features.to(device)
            labels = labels.to(device)

            # 1. forward
            logits = model(features)

            # 2. compute objective function (softmax, cross entropy)
            cost = loss_func(logits, labels)

            # 3. cleaning gradients
            # Otherwise, the gradient would be a combination of the old gradient, which you have already used to update your model parameters, and the newly-computed gradient. 
            optimizer.zero_grad()

            # 4. accumulate partial derivatives
            # When you call loss.backward(), all it does is compute gradient of loss w.r.t all the parameters in loss that have requires_grad = True and store them in parameter.grad attribute for every parameter. optimizer.step() updates all the parameters based on parameter.grad
            cost.backward()

            # 5. step in the opposite direction of the gradient
            optimizer.step()

            if not batch_index % 250:
                print('Epoch: {}/{} | Batch {}/{} | Cost: {:.4f}'.format(
                    epoch+1,
                    epochs,
                    batch_index,
                    len(train_loader),
                    cost
                ))

        correct, wrong, accuracy = comp_accuracy(model, train_loader, device)
        print('Training: Correct[{:.0f}] | Wrong[{:.0f}] | Accuracy[{:.2f}%]'.format(
            correct,
            wrong,
            accuracy
        ), '\n')

    end = time.time()
    print('Training time: {:.2f} seconds on {}'.format(
        end - start,
        device
    ))
