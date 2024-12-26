import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from model import GraphSAGEModel

graph_path = 'data/graph_data.pt'
data = torch.load(graph_path)

for node_type in data.node_types:
    num_nodes = data[node_type].num_nodes
    data[node_type].train_mask = torch.rand(num_nodes) < 0.8
    data[node_type].val_mask = (torch.rand(num_nodes) >= 0.8) & (torch.rand(num_nodes) < 0.9)
    data[node_type].test_mask = torch.rand(num_nodes) >= 0.9
    
train_loader = NeighborLoader(
    data,
    num_neighbors={key: [15, 10] for key in data.edge_types},  # Sample up to 15 and 10 neighbors for 2 layers
    batch_size=64,
    input_nodes=('EOA Address', data['EOA Address'].train_mask),  # Input nodes: EOA Address with train_mask
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_features = data['EOA Address'].x.size(1)  # Input feature size (e.g., 16)
model = GraphSAGEModel(
    metadata=data.metadata(), 
    in_channels=num_features,  # Match the number of node features
    hidden_channels=32,  # Hidden layer size
    out_channels=2,  # Binary classification
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
print("Starting training...")
for epoch in range(10):  # Number of epochs
    model.train()  # Set the model to training mode
    total_loss = 0

    for batch in train_loader:
        # Move batch to device
        batch = batch.to(device)

        # Forward pass
        out = model(batch.x_dict, batch.edge_index_dict)

        # Compute loss (cross-entropy for classification)
        loss = F.cross_entropy(
            out['EOA Address'][batch['EOA Address'].train_mask],  # Predictions for training nodes
            batch['EOA Address'].y[batch['EOA Address'].train_mask],  # Corresponding ground-truth labels
        )

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
    
torch.save(model.state_dict(), 'model_weights.pth')
print("Training completed. Model weights saved to 'model_weights.pth'.")
    
