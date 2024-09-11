import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import timm

class CustomDataset(Dataset):
    def __init__(self, root, metadata, train, num_channels, transform=None):
        self.metadata = metadata
        self.root = root
        self.train = train
        self.transform = transform
        self.num_channels = num_channels

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        image_filename = self.metadata["ID"][index]

        image = Image.open(
            os.path.join(self.root, image_filename + ".jpg"), formats=["JPEG"]
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        image = image[:self.num_channels]

        if self.train:
            label = self.metadata["CLASS"][index]
            return image, label

        return image
    
metadata_path = 'metadata/metadataTrain.csv'

metadata_train = pd.read_csv(metadata_path)
metadata_train['CLASS'] -= 1

train_dir = 'Train'
metadata_file = 'metadata/metadataTest.csv'

img_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the dataset
train_data = CustomDataset(train_dir, metadata_train, train=True, num_channels=3, transform=img_transform)

# separate into train and validation
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size], generator=torch.Generator().manual_seed(0))

# check GPU to work
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print("Device: ", device)

model = timm.create_model('inception_resnet_v2', pretrained=False, num_classes=8)

learning_rate = 0.0001
num_epochs = 20
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(0), num_workers=os.cpu_count())
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

# train model
model.to(device)
class_weights = [0.7005531, 0.24592265, 0.95261733, 3.64804147, 1.20674543, 13.19375, 12.56547619, 5.04219745]
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

# Initialize lists to store training and validation metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

best_val_accuracy = 0.0  # Initialize best validation accuracy

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    epoch_train_loss = 0.0
    correct_train = 0
    total_train = 0

    # Training phase with tqdm progress bar
    with tqdm(train_loader, desc=f'Train Epoch {epoch + 1}/{num_epochs}', unit='batch') as train_tqdm:
        for images, labels in train_tqdm:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optimizer.step()
            epoch_train_loss += train_loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            train_tqdm.set_postfix({'train_loss': train_loss.item(), 'train_accuracy': 100 * correct_train / total_train})
    
    # Calculate average training loss and accuracy for the epoch
    avg_train_loss = epoch_train_loss / len(train_loader)
    avg_train_accuracy = 100 * correct_train / total_train
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_accuracy)

    # Validation phase
    model.eval()  # Set model to evaluation mode
    epoch_val_loss = 0.0
    correct_val = 0
    total_val = 0

    # Disable gradients during validation
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            val_outputs = model(val_images)
            val_loss = criterion(val_outputs, val_labels)
            epoch_val_loss += val_loss.item()

            # Calculate accuracy
            _, predicted = torch.max(val_outputs, 1)
            total_val += val_labels.size(0)
            correct_val += (predicted == val_labels).sum().item()
            
    # Calculate average validation loss and accuracy for the epoch
    avg_val_loss = epoch_val_loss / len(val_loader)
    avg_val_accuracy = 100 * correct_val / total_val
    val_losses.append(avg_val_loss)
    val_accuracies.append(avg_val_accuracy)
    
    # Update the scheduler according to validation loss
    scheduler.step(avg_val_loss)

    # Display training and validation loss, accuracy for the epoch
    current_lr = optimizer.param_groups[0]['lr']  # Get current learning rate
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.2f}%, LR: {current_lr}')
    
    # Check if current validation accuracy is better than the best so far
    if avg_val_accuracy > best_val_accuracy:
        best_val_accuracy = avg_val_accuracy
        # Save the model checkpoint with the best validation accuracy
        torch.save(model.state_dict(), 'classification_resnet_1.pth')

# Saving data to a file
file_path = "resnet_metrics1.txt"
with open(file_path, "w") as file:
    file.write("Epoch\tTrain Loss\tVal Loss\tTrain Accuracy\tVal Accuracy\n")
    for epoch in range(num_epochs):
        file.write(f"{epoch+1}\t{train_losses[epoch]}\t{val_losses[epoch]}\t{train_accuracies[epoch]}\t{val_accuracies[epoch]}\n")

print("Training metrics saved to:", file_path)