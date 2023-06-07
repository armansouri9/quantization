# Quantization Repository

This repository contains a Python code example for applying quantization to a neural network model using PyTorch.

## Repository Specifications

- Repository URL: [https://github.com/armansouri9/quantization](https://github.com/armansouri9/quantization)

## Code

The following code demonstrates how to quantize a neural network model using PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = self.fc(x)
        return x

# Create an instance of the model
model = Net()

# Set the model to training mode and train it

# Set the model to evaluation mode
model.eval()

# Set the path to save the model weights
model_path = 'model.pth'

# Save the model weights
torch.save(model.state_dict(), model_path)

# Load the model weights
loaded_model = Net()
loaded_model.load_state_dict(torch.load(model_path))

# Apply quantization to the model
quantized_model = torch.quantization.quantize_dynamic(loaded_model, {torch.nn.Linear}, dtype=torch.qint8)

# Compare the performance of the model before and after quantization

# Define the dataset used for model evaluation
data = ...

# Evaluate the model before quantization
output_before_quantization = loaded_model(data)

# Evaluate the model after quantization
output_after_quantization = quantized_model(data)

# Compare the output of the model before and after quantization
print("Output before quantization:", output_before_quantization)
print("Output after quantization:", output_after_quantization)
```

Feel free to explore the repository for more information and additional files.

Please let me know if you need any further assistance!

## License

This project is licensed under a Free License.
