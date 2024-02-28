import torch
import torch.nn as nn
import numpy as np

"""
Input and Output for a hooked layer
"""
def get_values_hook(module, inp, out):
    model.layer_id += 1

    # get the name of module as string format
    name_module = str(type(module))
    print('Layer ID: {} || module: {}'.format(model.layer_id, name_module))

    # save input values as a txt file
    if isinstance(inp, tuple):
        inp = inp[0]
    _, Channel, S, _  = inp.shape
    reshaped_input = inp.reshape(Channel, -1) # 4D to 2D (Channel, S*S)
    print('\t*input* reshaped_input: {}'.format(reshaped_input.shape))
    np.savetxt('./input_ID['+str(model.layer_id)+']-['+name_module+'].txt', reshaped_input.detach().numpy(), delimiter=',')

    # save output values as a txt file
    _, Channel, S, _  = out.shape
    reshaped_output = out.reshape(Channel, -1) # 4D to 2D (Channel, S*S)
    print('\t*output* reshaped_output: {}'.format(reshaped_output.shape))
    np.savetxt('./output_ID['+str(model.layer_id)+']-['+name_module+'].txt', reshaped_output.detach().numpy(), delimiter=',')

    print()


"""
Define your model
"""
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        # Define the layers of the CNN
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 16 * 16, 256)  # Assuming input image size 32x32
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Define the forward pass
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        return x


"""
Create an instance of the CNN model
"""
model = CNN()


"""
Register the hook function
"""
for name, module in model.named_modules():

    # if you want to hook on Convolution layers
    if 'Conv' in str(type(module)):
        module.register_forward_hook(get_values_hook)

    """
    # if you want to hook on FC layers
    if 'Linear' in str(type(module)):
        module.register_forward_hook(get_values_hook)
    """
    
    """
    # if you want to hook on all layers.
    module.register_forward_hook(get_values_hook)
    """
    


example_input = torch.normal(0, 1, size=(1, 3, 32, 32))
model.layer_id = 0
model(example_input)