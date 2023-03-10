import torch
import matplotlib.pyplot as plt
from torchvision import transforms as T
from PIL import Image
from hook import OutputHook
from hook import tiny_VGG


class Visualize_Feature_Map():
    def __init__(self, model, image_path):
        self.hook = OutputHook()
        self.hook_handles = []
        for layer in model.modules():
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                handle = layer.register_forward_hook(self.hook)
                self.hook_handles.append(handle)
        image = Image.open(image_path)
        transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        x = transform(image).unsqueeze(dim=0)
        out = model(x)

    def visualize_one_layer(self, target_layer):
        feature_maps = []
        for i in range(len(self.hook_handles)):
            images = self.hook.outputs[i]
            feature_maps.append(images)
        processed = []
        for feature_map in feature_maps:
            feature_map = feature_map.squeeze(0)
            gray_scale = torch.sum(feature_map, 0)
            gray_scale = gray_scale / feature_map.shape[0]
            processed.append(gray_scale.data.cpu().numpy())
        plt.figure(figsize=(20, 20))
        plt.title(f"The feature map of layer {target_layer}")
        plt.imshow(processed[target_layer-1])
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        plt.show()

    def visualize_all_layer(self):
        feature_maps = []
        for i in range(len(self.hook_handles)):
            images = self.hook.outputs[i]
            feature_maps.append(images)
        processed = []
        for feature_map in feature_maps:
            feature_map = feature_map.squeeze(0)
            gray_scale = torch.sum(feature_map, 0)
            gray_scale = gray_scale / feature_map.shape[0]
            processed.append(gray_scale.data.cpu().numpy())

        plt.figure(figsize=(20, 20))
        n_figure = len(processed)
        n_col = 2
        if (n_figure % 2 == 0):
            n_row = n_figure // n_col
        else:
            n_row = (n_figure//n_col) + 1
        for idx in range(4):
            plt.subplot(n_col, n_row, idx + 1)
            plt.imshow(processed[idx])
            plt.title(f"The feature map of layer {idx+1}")
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        plt.show()


if __name__ == "__main__":
    model = tiny_VGG(2)
    image_path = 'f4bffa3d-e4bf-49dd-b222-99834ef964f0.jpeg'
    feature = Visualize_Feature_Map(model, image_path)
    feature.visualize_all_layer()
    feature.visualize_one_layer(2)
