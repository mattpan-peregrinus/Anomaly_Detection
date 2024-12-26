import torch
from model import GraphSAGEModel

graph_path = 'data/graph_data.pt'
model_path = 'model_weights.pth'
data = torch.load(graph_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGEModel(metadata=data.metadata(), in_channels=16, hidden_channels=32, out_channels=2).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

test_mask = data['EOA Address'].test_mask
out = model(data.x_dict.to(device), data.edge_index_dict.to(device))['EOA Address']
pred = out.argmax(dim=1)
correct = (pred[test_mask] == data['EOA Address'].y[test_mask]).sum().item()
accuracy = correct / test_mask.sum().item()

print(f"Test Accuracy: {accuracy:.4f}")