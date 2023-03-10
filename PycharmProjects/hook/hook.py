import torch.nn as nn



class tiny_VGG(nn.Module):
    def __init__(self,num_class):
        super(tiny_VGG, self).__init__()
        self.layer_1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=10,kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(10,10,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.layer_2=nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classify=nn.Sequential(
            nn.Flatten(),
            nn.Linear(28090,num_class)
        )
    def forward(self,x):
        out=self.layer_1(x)
        out=self.layer_2(out)
        out=self.classify(out)
        return out

class OutputHook:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []



