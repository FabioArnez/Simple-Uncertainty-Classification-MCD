import torch
import torch.nn as nn
import torch.nn.functional as F

# num epochs
# epochs = 20
# learning rate
# learning_rate = 0.001

def TrainModel(model, device, train_loader, lossFunction, optimizer):
    sum_traning_batches_avg_loss = 0.0
    # Enable train mode
    model.train()

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = lossFunction(output, target)
        loss.backward()
        optimizer.step()
        # update running training loss
        sum_traning_batches_avg_loss += loss.item() # sum up batch loss

    return sum_traning_batches_avg_loss


def ValidateModel(model, device, valid_loader, lossFunction):
    sum_valid_batches_avg_loss = 0.0
    # Enable train mode
    model.eval()

    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = lossFunction(output, target)
            # update running evaluation loss
            sum_valid_batches_avg_loss += loss.item()  # sum up batch loss

    return sum_valid_batches_avg_loss


def TrainValidNeuralNet(model, train_loader, valid_loader, num_epochs):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # send nn model to GPU
    model.to(device)

    # specify loss function (categorical cross-entropy)
    # lossFunction = nn.CrossEntropyLoss()
    lossFunction = nn.NLLLoss()
    # specify optimizer
    learning_rate = 0.01
    # specify optimizer (stochastic gradient descent) and learning rate = 0.01
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epoch_avg_train_losses, epoch_avg_eval_losses = [], []

    for epoch in range(1, num_epochs + 1):
        sum_traning_batches_avg_loss = 0.0
        sum_valid_batches_avg_loss = 0.0

        ###################
        # train the model #
        ###################
        sum_traning_batches_avg_loss = TrainModel(model, device, train_loader, lossFunction, optimizer)
        ######################    
        # validate the model #
        ######################
        sum_valid_batches_avg_loss = ValidateModel(model, device, valid_loader, lossFunction)

        # Get epoch average loss
        epoch_avg_train_loss = sum_traning_batches_avg_loss / len(train_loader)
        epoch_avg_valid_loss = sum_valid_batches_avg_loss / len(valid_loader)

        # append epoch average losses
        epoch_avg_train_losses.append(epoch_avg_train_loss)
        epoch_avg_eval_losses.append(epoch_avg_valid_loss)

        print("Epoch: {}/{}.. ".format(epoch, num_epochs),
          "Avg. Training Loss: {:.3f}.. ".format(epoch_avg_train_loss),
          "Avg. Validation Loss: {:.3f}.. ".format(epoch_avg_valid_loss))

    return epoch_avg_train_losses, epoch_avg_eval_losses
