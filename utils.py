import torch
from torch_geometric.nn import GCNConv, BatchNorm
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Define a simple GCN model
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCN_with_dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(GCN_with_dropout, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout_rate = dropout_rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)



class GCN_with_dropout_and_bn(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(GCN_with_dropout_and_bn, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)  # Batch normalization layer after conv1
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.bn2 = BatchNorm(output_dim)  # Batch normalization layer after conv2
        self.dropout_rate = dropout_rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)  # Apply batch normalization
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)  # Apply batch normalization
        x = F.dropout(x, p=self.dropout_rate, training=self.training)  # Optional: Dropout before softmax
        
        return F.log_softmax(x, dim=1)

# # Define the GCN model
# class GCN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, output_dim)

#     def forward(self, x, edge_index):
#         x = F.relu(self.conv1(x, edge_index))
#         x = self.conv2(x, edge_index)
#         return x


def generate_train_test(data, split_ratio=0.8):
    # Define train and test masks
    num_nodes = data.num_nodes
    num_train = int(num_nodes * 0.8)  # For example, 80% for training
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    # Randomly select nodes for training and testing
    perm = torch.randperm(num_nodes)
    train_mask[perm[:num_train]] = True
    test_mask[perm[num_train:]] = True
    
    # Assign masks to data
    data.train_mask = train_mask
    data.test_mask = test_mask

    # Filter edges for train_pos_edge_index
    train_nodes = perm[:num_train].tolist()
    edge_index = data.edge_index
    train_edge_mask = train_mask[edge_index[0]] & train_mask[edge_index[1]]
    data.train_pos_edge_index = edge_index[:, train_edge_mask]


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, data):
    model.eval()
    logits, accs = model(data), []
    for mask in [data.train_mask, data.test_mask]:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def evaluate_model(model, data, gae=False):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        preds = logits.argmax(dim=1)
        labels = data.y

    # Calculate metrics only for the test set
    mask = data.test_mask
    test_preds = preds[mask].cpu().numpy()
    test_labels = labels[mask].cpu().numpy()

    accuracy = accuracy_score(test_labels, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='macro')

    return accuracy, precision, recall, f1



def train_loop(n_epochs, model, optimizer, data, acc_freq=5, verbose=False):
    # Initialize lists to store the metrics
    loss_history = []
    train_acc_history = []
    test_acc_history = []
    
    for epoch in range(200):
        loss = train(model, optimizer, data)
        loss_history.append(loss)
        
        # Optionally, calculate training accuracy every few epochs to save computation
        if epoch % acc_freq == 0 and acc_freq != -1:
            train_acc, test_acc = test(model, data)
            train_acc_history.append(train_acc)
            test_acc_history.append(test_acc)
            if verbose:
                print(f'Epoch {epoch+1}, Loss: {loss:.4f}, Training Accuracy: {train_acc:.4f}')
                
    return loss_history, train_acc_history, test_acc_history



def plot_training_history(loss_history, train_acc_history, test_acc_history, acc_freq=5, grid=False):
    # Plotting the loss history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training loss history')
    plt.legend()
    # if grid:
    plt.grid(grid)
    
    # Plotting the training accuracy history
    plt.subplot(1, 2, 2)
    epochs_acc = range(0, len(loss_history), int(len(loss_history)/len(train_acc_history)))
    plt.plot(epochs_acc, train_acc_history, label='Training accuracy', color='orange')
    plt.plot(epochs_acc, test_acc_history, label='Test accuracy', color='darkred')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and test accuracy history')
    plt.legend()
    # if grid:
    plt.grid(grid)
    
    plt.tight_layout()
    plt.show()


def plot_no_show(ax, results, labels, title, size=None, sample=False):
    unique_labels = np.unique(labels)
    if sample:
        sample_fraction=0.3
    else:
        sample_fraction = None
    if sample_fraction is not None:
        # Determine the number of samples to take
        num_samples = int(len(results) * sample_fraction)
        # Create a random choice of indices based on the sample size
        sampled_indices = np.random.choice(len(results), num_samples, replace=False)
        # Subset results and labels
        results = results[sampled_indices]
        labels = labels[sampled_indices]
    
    # for label in unique_labels:
    #     indices = labels == label
    #     ax.scatter(results[indices, 0], results[indices, 1], label=str(label), alpha=0.1, s=size)
    color_dict = {0: u'#1f77b4', 1: u'#ff7f0e', 2: u'#2ca02c', 3: u'#d62728', 4: u'#9467bd', 5: u'#8c564b', 6: u'#e377c2', 7: u'#7f7f7f', 8: u'#bcbd22', 9: u'#17becf'}
    ax.scatter(results[:, 0], results[:, 1], c=[color_dict[label] for label in labels], s=size)
    # ax.set_xlabel('Feature 1')
    # ax.set_ylabel('Feature 2')
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(title)


def visualise(results, labels, title, size=None, sample=False):
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_no_show(ax, results, labels, title, size, sample)
    plt.show()