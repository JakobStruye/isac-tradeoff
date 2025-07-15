import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, num_classes=8,out_channels_1=16,out_channels_2=32,out_channels_3=64,kernel_size_1=3,kernel_size_2=3,kernel_size_3=7, shape=(50,56)):
        super(ConvNet, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=shape[0], out_channels=out_channels_1, kernel_size=kernel_size_1, padding=1),
            nn.BatchNorm2d(out_channels_1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=out_channels_1, out_channels=out_channels_2, kernel_size=kernel_size_2, padding=1),
            nn.BatchNorm2d(out_channels_2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=out_channels_2, out_channels=out_channels_3, kernel_size=kernel_size_3, padding=1),
            nn.BatchNorm2d(out_channels_3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate the input size for the linear layer dynamically
        self.conv_output_size = self._get_conv_output_size((shape[0], shape[1], shape[2]))  # Example input shape (1, 30, 50)
        self.linear_layer = nn.Sequential(nn.Dropout(p=0),
                                          nn.Linear(self.conv_output_size, num_classes))
        
    def _get_conv_output_size(self, shape):
        batch_size = 1
        input_tensor = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._forward_features(input_tensor)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.conv_layers(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layer(x)
        return x