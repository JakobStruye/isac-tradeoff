import sys
import math
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.classification import MulticlassConfusionMatrix

from convnet import ConvNet

def parse_args():
    parser = argparse.ArgumentParser(description='Qamcom dataset')
    parser.add_argument('--seed', type=int, default=999999, help='The random seed')
    parser.add_argument('--num_epochs', type=int, default=1, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='The batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='The learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='The weight decay regularizer for Adam optimizer')
    parser.add_argument('--beta1', type=float, default=0.9, help='The beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='The beta2 for Adam optimizer')
    parser.add_argument('--eps', type=float, default=1e-10, help='The epsilon for Adam optimizer')
    parser.add_argument('--patience', type=int, default=25,
                        help='The patience for ReduceLROnPlateau scheduler')

    parser.add_argument('--subsample_tx_beams', type=int, default=1, help='Tx beam subsample rate')
    parser.add_argument('--subsample_rx_beams', type=int, default=1, help='Rx beam subsample rate')
    parser.add_argument('--subsample_time', type=int, default=1, help='Temporal subsample rate')
    parser.add_argument('--subsample_approach', choices=["repeated", "interleaved"], default="interleaved", help="The subsampling method")
    parser.add_argument('--scramble_data', action=argparse.BooleanOptionalAction, help='Scramble beam order')

    parser.add_argument('--log_to_file', action=argparse.BooleanOptionalAction, help='Log to file')
    args = parser.parse_args()
    return args

def  train_test_model(net, train_data_loader,  test_data_loader,optimizer, criterion, scheduler1, num_epochs, device,checkpoint):
    best_test_accuracy=0.0
    train_loss = []

    for epoch in range(num_epochs):
        sigma=torch.tensor(random.uniform(0, 0.5)).to(device)
        print(sigma)
        net.train()
        running_loss = 0.0
        for (samples, labels) in tqdm(train_data_loader):
            samples = samples.to(torch.float32)
            samples =samples.to(device)
            samples = Variable(samples.to(device))
            labels = labels.squeeze()
            labels = Variable(labels.to(device))

            optimizer.zero_grad()

            predict_output = net(samples)
            loss = criterion(predict_output, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_loss.append(running_loss / len(train_data_loader))
        scheduler1.step(train_loss[-1])

        print(f"\nTrain loss for epoch {epoch+1} is {train_loss[-1]:.3f}")

        net.eval()
      
        test_loss = 0
        correct_test = 0
        for samples, labels in test_data_loader:
            with torch.no_grad():
                samples = Variable(samples.to(device))
                labels = labels.squeeze()
                labels = Variable(labels.to(device))

                predict_output = net(samples)
                predicted_label = torch.max(predict_output, 1)[1]

                correct_test += (predicted_label == labels).sum().item()
                loss = criterion(predict_output, labels)
                test_loss += loss.item()
        test_loss /= len(test_data_loader)
        test_acc = 100 * float(correct_test) / len(test_data_loader.dataset)
        if  test_acc > best_test_accuracy:
             # Save the model state
             print(".....saving model...")
             best_test_accuracy = test_acc
             torch.save(checkpoint,  filename + ".pth")
             print("best test accuracy is",best_test_accuracy)
        print("Test accuracy:", test_acc)
        print(f"Test loss is {test_loss:.3f}")

    return best_test_accuracy


def custom_split_tensor(tensor, block_starts, train, validation): #test is implicit

    # List to store the three tensors for each part
    first_part = []
    second_part = []
    third_part = []

    # Process each block
    for i in range(len(block_starts) - 1):  # Last block is from block_starts[-1] to the end
        start_idx = block_starts[i]
        end_idx = block_starts[i + 1] if i + 1 < len(block_starts) else len(tensor)

        # Calculate the number of elements in this block
        block_length = end_idx - start_idx

        # Calculate split points for each block
        first_split = int(block_length * train )
        second_split = int(block_length * validation)

        # Split the block
        first_part.append(tensor[start_idx:start_idx + first_split])
        second_part.append(tensor[start_idx + first_split:start_idx + first_split + second_split])
        third_part.append(tensor[start_idx + first_split + second_split:end_idx])

    # Convert the lists of tensors into one tensor for each part
    first_part = torch.cat(first_part)
    second_part = torch.cat(second_part)
    third_part = torch.cat(third_part)

    return first_part, second_part, third_part


def beam_scramble(data):

    num_samples, time_steps, height, width = data.shape
    scrambled_data = data.clone()  # Copy to avoid modifying original data

    # Generate a single random permutation for the entire dataset
    perm_indices = torch.randperm(height * width, device=data.device)  # Flattened indices
    perm_indices = perm_indices.view(height, width)  # Reshape back to (50, 56)

    # Apply this permutation to all samples and all time steps
    for sample_idx in range(num_samples):
        for t in range(time_steps):
            scrambled_data[sample_idx, t] = data[sample_idx, t].flatten()[perm_indices].view(height, width)

    return scrambled_data


def compute_mean_spatial_saliency(model, X_test, num_samples=100, target_class=0):
    model.eval()  # Set model to evaluation mode

    # Shuffle test indices
    num_test_samples = len(X_test)
    shuffled_indices = torch.randperm(num_test_samples)  # Random permutation of indices
    selected_indices = shuffled_indices[:num_samples]  # Select first num_samples after shuffle

    total_saliency = None
    count = 0

    for idx in selected_indices:
        input_tensor = X_test[idx].clone().detach().to(device).requires_grad_(True)  # Enable gradients
        input_tensor_shaped = input_tensor.unsqueeze(0)
        # Forward pass
        output = model(input_tensor_shaped)  # Add batch dim
        loss = output[0, target_class]  # Select target class output
        loss.backward()  # Backpropagate

        # Compute saliency for this sample (50, 56 map)
        saliency_map = input_tensor.grad.abs().mean(dim=0)  # Aggregate over time

        # Accumulate saliency maps
        if total_saliency is None:
            total_saliency = saliency_map
        else:
            total_saliency += saliency_map

        count += 1

    return total_saliency / count if count > 0 else None  # Normalize by sample count

def nearest_repeat_subsample(tensor, dim, factor):
    """
    Subsamples every `factor`-th element along `dim`, then repeats each selected
    value in blocks to reconstruct the original size along that dimension.
    Supports any tensor shape and any dim.
    """
    size = tensor.size(dim)
    device = tensor.device

    # Downsample indices along the chosen dim
    idx = torch.arange(0, size, factor, device=device)
    downsampled = torch.index_select(tensor, dim, idx)

    n_blocks = downsampled.size(dim)
    base_block = size // n_blocks
    remainder = size % n_blocks

    # Create list of repeat counts: distribute remainder by adding 1 to first blocks
    repeats = [base_block + 1 if i < remainder else base_block for i in range(n_blocks)]
    repeats_tensor = torch.tensor(repeats, device=device)

    # Repeat each block along dim
    expanded = downsampled.repeat_interleave(repeats_tensor, dim=dim)

    return expanded


def striated_subsample(tensor, dim, factor):
    """
    Subsamples every `factor`-th element along `dim`, then interleaves
    the selected values cyclically to fill the original size along `dim`.
    Supports arbitrary tensor shape and dimension.
    """
    size = tensor.size(dim)
    device = tensor.device

    # Downsampled indices and values
    idx = torch.arange(0, size, factor, device=device)
    downsampled = torch.index_select(tensor, dim, idx)

    n_down = downsampled.size(dim)

    # Compute how many repeats needed to cover original size
    n_repeats = (size + n_down - 1) // n_down  # ceil division

    # Prepare repeat sizes for all dims:
    repeats = [1] * tensor.ndim
    repeats[dim] = n_repeats

    # Repeat along `dim`
    tiled = downsampled.repeat(*repeats)

    # Narrow (slice) back to original size along `dim`
    tiled = tiled.narrow(dim, 0, size)

    return tiled


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Parse arguments
    args = parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    filename = "{}_epoch{}".format(args.network, args.num_epochs)


    data=torch.load("dataset_concat/all_data.pth")
    data = torch.cat((data[:8625],data[12940:])) #Remove unsorted samples
    
    labels=torch.load("dataset_concat/all_labels.pth")
    labels = torch.cat((labels[:8625], labels[12940:]))
    
    # print("Loaded")
    labels=labels.to(torch.int64)

    # Or subsample measurements
    if args.subsample_time > 1:
        data = nearest_repeat_subsample(data, 1, args.subsample_time)
        filename += "_subtime{}".format(args.subsample_time)

    #Subsample beams
    if args.subsample_tx_beams > 1:
        if args.subsample_approach == "repeated":
            data = nearest_repeat_subsample(data, 2, args.subsample_tx_beams)
        elif args.subsample_approach == "interleaved":
            data = striated_subsample(data, 2, args.subsample_tx_beams)
        filename += "_subtx{}".format(args.subsample_tx_beams)


    if args.subsample_rx_beams > 1:
        if args.subsample_approach == "repeated":
            data = nearest_repeat_subsample(data, 3, args.subsample_rx_beams)
        elif args.subsample_approach == "interleaved":
            data = striated_subsample(data, 3, args.subsample_rx_beams)
        filename += "_subrx{}".format(args.subsample_rx_beams)

    if args.subsample_tx_beams > 1 or args.subsample_rx_beams > 1:
        filename += "_{}".format(args.subsample_approach)

    # Scramble beam order within sample
    if args.scramble_data:
        data = beam_scramble(data)
        filename+= "_scramble"

    if args.seed != 999999:
        filename += "_seed{}".format(args.seed)

    if args.log_to_file:
        log_file = open(filename + ".log", "w")
        sys.stdout = log_file
        sys.stderr = log_file

    user_starts = torch.nonzero((labels == 0) & (torch.roll(labels, 1) != 0)).squeeze()
    gesture_starts = torch.nonzero(labels[1:] != labels[:-1]).squeeze() + 1

    X_train, X_val, X_test = custom_split_tensor(data, gesture_starts, 0.72, 0.08)
    y_train, y_val, y_test = custom_split_tensor(labels, gesture_starts, 0.72, 0.08)

    # # Define model architecture
    ####                           Dataloader
    train_dataset=TensorDataset(X_train,y_train)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=False)
    test_dataset = TensorDataset(X_test,y_test)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,drop_last=False)
    test_data_loader=test_data_loader
    val_dataset=TensorDataset(X_val,y_val)
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)

    
    print("n size of training data is", len(train_data_loader.dataset))
    print("n size of val data is", len(val_data_loader.dataset))

    print("size of test data is", len(test_data_loader.dataset))

    print("Data shape", data.shape)
    input_dim = math.prod(data.shape[1:])

    net=ConvNet(shape=data.shape[1:]).to(device).to(dtype=torch.float32)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08,weight_decay=args.weight_decay)
    
    scheduler1 = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.patience)
    checkpoint = {'model_state_dict': net.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  }      
    best_val_accuracy=train_test_model(net,  train_data_loader,  val_data_loader,optimizer,
        criterion, scheduler1, args.num_epochs, device,checkpoint)
    print("best validation accuracy is",best_val_accuracy)

    # Load the saved checkpoint
    checkpoint = torch.load(filename + '.pth')

    # Load the model state dict into the model
    net.load_state_dict(checkpoint['model_state_dict'])

    test_loss = 0
    correct_test = 0
    all_pred = torch.empty(0, device=device)
    all_truth = torch.empty(0, device=device)
    for i, (samples, labels) in enumerate(test_data_loader):
        with torch.no_grad():
            samples = Variable(samples.to(device))
            labels = labels.squeeze()
            labels = Variable(labels.to(device))

            predict_output = net(samples)
            predicted_label = torch.max(predict_output, 1)[1]
            all_pred = torch.cat([all_pred, predicted_label])
            all_truth = torch.cat([all_truth, labels])
            correct_test += (predicted_label == labels).sum().item()
            loss = criterion(predict_output, labels)
            test_loss += loss.item()

    test_loss /= len(test_data_loader)
    test_acc = 100 * float(correct_test) / len(test_data_loader.dataset)
    metric = MulticlassConfusionMatrix(num_classes=8).to(device)
    metric.update(all_pred, all_truth)
    fig_, ax_ = metric.plot()
    fig_.savefig(filename + '_confmat.png')

    print("Test accuracy:", test_acc)
    print(f"Test loss is {test_loss:.3f}")

    # Extract test data directly from dataset
    X_test = test_dataset.tensors[0]  # Assuming test_dataset = TensorDataset(X_test, y_test)

    # Compute mean saliency over 100 **random** test samples
    mean_saliency_map = compute_mean_spatial_saliency(net, X_test, num_samples=1000, target_class=0)

    # Visualizing the mean spatial importance heatmap
    if mean_saliency_map is not None:
        # plt.close()
        fig_.clf()
        plt.imshow(mean_saliency_map.detach().cpu().numpy(), cmap='hot', interpolation='nearest')
        cbar = plt.colorbar()
        # cbar.set_label("Beam pair saliency")
        plt.xlabel("Tx beam index")
        plt.ylabel("Rx beam index")
        plt.title("Beam pair importance heatmap")
        plt.savefig(filename + "_heatmap.png")
    else:
        print("No saliency map was computed.")

    if args.log_to_file:
        log_file.close()