import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load and preprocess image
def load_image(path, size=400):
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# Convert tensor to image
def tensor_to_image(tensor):
    image = tensor.squeeze().cpu().clone().detach()
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    image = image + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    return image.clamp(0,1).permute(1,2,0).numpy()

# Calculate Gram matrix (style)
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    return torch.mm(features, features.t())

# Main style transfer function
def run_style_transfer(content_path, style_path, steps=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load images
    content = load_image(content_path).to(device)
    style = load_image(style_path).to(device)
    target = content.clone().requires_grad_(True)

    # Load VGG19
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False

    # Layers for content and style
    content_layer = '21'  # relu4_2
    style_layers = ['0', '5', '10', '19', '28']

    def get_features(x):
        features = {}
        for name, layer in vgg._modules.items():
            x = layer(x)
            if name in [content_layer] + style_layers:
                features[name] = x
        return features

    # Extract features
    content_feat = get_features(content)[content_layer]
    style_feats = get_features(style)
    style_grams = {l: gram_matrix(style_feats[l]) for l in style_layers}

    # Optimizer
    optimizer = optim.Adam([target], lr=0.003)

    for i in range(steps):
        target_feats = get_features(target)
        c_loss = torch.mean((target_feats[content_layer] - content_feat) ** 2)

        s_loss = 0
        for l in style_layers:
            t_gram = gram_matrix(target_feats[l])
            s_gram = style_grams[l]
            s_loss += torch.mean((t_gram - s_gram) ** 2)

        total_loss = 1e4 * c_loss + 1e2 * s_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"Step {i} - Loss: {total_loss.item():.4f}")

    # Show final image
    output = tensor_to_image(target)
    plt.imshow(output)
    plt.axis('off')
    plt.title("Stylized Image")
    plt.show()

# ✨ Run with your images
run_style_transfer('your_photo.jpg', 'your_style.jpg')
