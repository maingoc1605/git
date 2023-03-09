import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torchvision import transforms as T
from PIL import Image


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


hook_handles = []
hook = OutputHook()
model = tiny_VGG(2)
for layer in model.modules():
    if isinstance(layer, torch.nn.modules.conv.Conv2d):
        handle = layer.register_forward_hook(hook)
        hook_handles.append(handle)

image = Image.open('f4bffa3d-e4bf-49dd-b222-99834ef964f0.jpeg')
transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
X = transform(image).unsqueeze(dim=0)
out = model(X)

feature_maps =[]

for i in range(len(hook_handles)):
    images = hook.outputs[i]
    feature_maps.append(images)

processed = []

for feature_map in feature_maps:
    feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map,0)
    gray_scale = gray_scale / feature_map.shape[0]
    processed.append(gray_scale.data.cpu().numpy())

plt.figure(figsize=(20, 20), frameon=False)
for idx in range(4):
    plt.subplot(2, 2, idx+1)
    plt.imshow(processed[idx])
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.show()
