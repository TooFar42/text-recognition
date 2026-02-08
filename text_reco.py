import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import os
import string
from PIL import Image
from tqdm import tqdm # Import tqdm for the progress bar

# --- 1. CONFIGURATION ---
DATA_DIR = "data"
BATCH_SIZE = 64
LEARNING_RATE = 0.0003
EPOCHS = 15
IMG_SIZE = (128,128)
ALPHABET = string.ascii_uppercase + string.digits
NUM_CLASSES = len(ALPHABET) + 1 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. DATASET CLASS ---
class OCRDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.img_paths = [i for i in os.listdir(data_dir) if i.endswith(".png")]
        self.transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        # Mapping: Blank is 0, 'A' is 1, etc.
        self.char_to_int = {char: i + 1 for i, char in enumerate(ALPHABET)}

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_filename = self.img_paths[index]
        # Label extraction from "index_LABEL.png"
        label_str = img_filename.split("_")[-1].split(".")[0]
        label_int = [self.char_to_int[char] for char in label_str]
        
        full_path = os.path.join(self.data_dir, img_filename)
        image = Image.open(full_path).convert('L')
        image = self.transform(image)
        
        return image, torch.LongTensor(label_int), torch.tensor(len(label_int))

def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images = torch.stack(images, 0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    lengths = torch.stack(lengths)
    return images, labels_padded, lengths

# --- 3. THE AI MODEL (CRNN) ---
class AImodel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),      # 64
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 32
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d((4, 2)), # H:4, W:8
        )
        # 1024 = 256 channels * 4 height
        self.rnn = nn.LSTM(1024, 256, bidirectional=True, num_layers=2, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous() # Width becomes the "time" sequence
        x = x.view(b, w, c * h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x.log_softmax(2)

# --- 4. TRAINING LOOP ---
def train():
    dataset = OCRDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    model = AImodel(NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    print(f"Starting training on {device} with {len(dataset)} samples...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # Wrap the DataLoader with tqdm
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        
        for images, labels, label_lengths in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward
            preds = model(images)  # [Batch, Time, Classes]
            
            # CTC expects: [Time, Batch, Classes]
            preds_ctc = preds.permute(1, 0, 2)
            
            # input_lengths is the width of the feature map (14 in this CNN)
            batch_size = images.size(0)
            input_lengths = torch.full((batch_size,), 8, dtype=torch.long).to(device)
            
            loss = criterion(preds_ctc, labels, input_lengths, label_lengths)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update the progress bar description with the current average loss
            avg_loss = total_loss / (progress_bar.n + 1)
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

        final_avg_loss = total_loss / len(loader)
        print(f"\nEpoch {epoch+1}/{EPOCHS} finished. Avg Loss: {final_avg_loss:.4f}")
    
    # Save the model
    torch.save(model.state_dict(), "ocr_model.pth")
    print("Model saved to ocr_model.pth")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"Error: folder '{DATA_DIR}' not found. Run your generator script first!")
    else:
        train()