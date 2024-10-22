import torch
from torch import nn
from torch.utils import data
from torchvision.transforms import Compose
from torchvision.models import resnet152
import os
import fdc
import transforms_nyu
from dataset import NyuV2
from den_gen2 import DEN  # Assuming den_gen2.py has the DEN class

# Path to the model weights directory
model_dir = './models/pretrained_resnet/'
os.makedirs(model_dir, exist_ok=True)

# Function to download and modify ResNet152 model
def download_and_modify_resnet():
    # Load pretrained ResNet152 model
    print("Downloading and modifying ResNet152 model...")
    resnet = resnet152(pretrained=True)
    
    # Modify the fully connected layer to match the required output size (800 in this case)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 800)  # Modify this if you need a different output size
    
    # Save the modified model's weights
    resnet_wts_path = os.path.join(model_dir, 'model.pt')
    torch.save(resnet.state_dict(), resnet_wts_path)
    print(f"Model saved to {resnet_wts_path}")
    return resnet_wts_path

# Call function to download and modify ResNet model
resnet_wts = download_and_modify_resnet()

# Continue with the rest of the pipeline
print("Setting up the DEN pipeline...")

# Define paths and configurations
data_path = './data/nyu_v2/'
seed = 2
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

depth_size = (25, 32)
model_input = 224
test_crop = (427, 561)
crop_ratios = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

# Define data transformations
transform = Compose([
    transforms_nyu.Normalize(),
    transforms_nyu.FDCPreprocess(crop_ratios)
])

# Create dataset and dataloader
nyu = NyuV2(os.path.join(data_path, 'train'), transform=transform)
dataloader = data.DataLoader(nyu, batch_size=1, shuffle=True, num_workers=6)

# Load the modified ResNet weights and initialize DEN
print("Loading DEN model with the modified ResNet weights...")
den = DEN()
den.load_state_dict(torch.load(resnet_wts))  # Load the pretrained ResNet weights into DEN
den = den.to(device)
den.eval()
print("DEN has been successfully loaded")

# Initialize FDC model and perform forward pass
fdc_model = fdc.FDC(den)
f_m_hat, f = fdc_model.forward(dataloader)

# Fit the FDC model and save the weights
fdc_model.fit(f_m_hat, f)
fdc_model.save_weights('./models/FDC/den_dbe/')

print("FDC model has been trained and weights saved.")
