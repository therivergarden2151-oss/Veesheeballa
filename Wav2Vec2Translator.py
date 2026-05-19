import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchaudio
import torchaudio.functional as F
from transformers import Wav2Vec2Config, Wav2Vec2ForSequenceClassification, get_scheduler, Wav2Vec2Processor
from tqdm import tqdm
import numpy as np
import os
import random # Used for setting seeds

# --- CONFIGURATION ---
NUM_CLASSES = 7 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "facebook/wav2vec2-base"

# Training Settings
LEARNING_RATE = 1.46e-05 # This seems like a good rate from my tests
N_EPOCHS = 5 		 # Example: I keep it small for test runs to save time
BATCH_SIZE = 4
# CLASS_WEIGHTS is already a tensor, defined here once.
# Using your provided weights for this run:
CLASS_WEIGHTS = torch.tensor([
    0.91100, # 0: veesheeballa 
    0.77000, # 1: otter 
    0.93000, # 2: window
    0.62000, # 3: squirrel
    1.09200, # 4: pantera
    1.09750, # 5: groundhog
    0.710000, # 6: armadillo
], dtype=torch.float32)

# --- USER'S DATA CONFIGURATION ---
TARGET_SAMPLE_RATE = 16000
TRAIN_DIR = r"C:\Users\Valued\Desktop\Dog Translator 2.0\Learning and Testing\Focus on Best Test\Learning"
TEST_DIR = r"C:\Users\Valued\Desktop\Dog Translator 2.0\Learning and Testing\Focus on Best Test\testing"

# Set Seeds for Reproducibility
def set_seeds(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
set_seeds()

# ===== HELPER FUNCTION: Print label maps for clarity =====
def get_label_name_map(label_map):
    """Inverts the label map to go from index to name."""
    return {v: k for k, v in label_map.items()}

# ===== USER'S DATASET CLASS (DogVoiceDataset) =====
class DogVoiceDataset(Dataset):
    def __init__(self, data_dir, is_test=False):
        self.data_dir = data_dir
        self.is_test = is_test
        self.samples = []

        # Training data folders are assumed to be named "0", "1", "2", "3", "4", "5", "6"
        self.label_map = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6}
        self.test_label_map = {
            "veesheeballa": 0,
            "otter": 1,
            "window": 2,
            "squirrel": 3,
            "pantera": 4,
            "groundhog": 5,
            "armadillo": 6
        }
        # Define the index-to-name map
        self.index_to_name = get_label_name_map(self.test_label_map) if self.is_test else get_label_name_map(self.label_map)

        if self.is_test:
            self._load_test_data()
        else:
            self._load_training_data()

    def _load_training_data(self):
        for label_name in os.listdir(self.data_dir):
            label_dir = os.path.join(self.data_dir, label_name)
            if os.path.isdir(label_dir) and label_name in self.label_map:
                label = self.label_map[label_name]
                for file in os.listdir(label_dir):
                    if file.endswith(".wav"):
                        filepath = os.path.join(label_dir, file)
                        self.samples.append((filepath, label))
            elif os.path.isdir(label_dir):
                print(f"Warning: Skipping folder '{label_name}' in training data. Not in label map.")

    def _load_test_data(self):
        # Store file path along with label for error reporting
        for file in os.listdir(self.data_dir):
            if file.endswith(".wav"):
                # Assumes file name starts with the label name followed by an underscore
                label_name = file.split("_")[0].lower()
                label = self.test_label_map.get(label_name)
                
                if label is not None:
                    filepath = os.path.join(self.data_dir, file)
                    self.samples.append((filepath, label))
                else:
                    print(f"Warning: Skipping test file '{file}'. Label '{label_name}' not in test label map.")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        try:
            waveform, sample_rate = torchaudio.load(filepath)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            # Return a tuple with None to indicate failure to load
            return None 

        if waveform.shape[0] > 1:
            # Convert stereo to mono by averaging channels
            waveform = waveform.mean(dim=0).unsqueeze(0)

        if sample_rate != TARGET_SAMPLE_RATE:
            # Resample if sample rate doesn't match the target
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)

        # Band-Pass Filtering
        waveform = F.highpass_biquad(waveform, sample_rate=TARGET_SAMPLE_RATE, cutoff_freq=150.0, Q=0.707)
        waveform = F.lowpass_biquad(waveform, sample_rate=TARGET_SAMPLE_RATE, cutoff_freq=2000.0, Q=0.707)
        
        waveform = torch.squeeze(waveform)

        return waveform, label, filepath # Return filepath here for logging

# ===== WAV2VEC2 PROCESSOR SETUP =====
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

config = Wav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES
).config

# Adjust mask length if needed (to prevent short-sample error)
if hasattr(config, 'mask_length') and config.mask_length >= 9:
    config.mask_length = 8
    print(f"🛠️ Config updated: mask_length set to {config.mask_length} to prevent short-sample error.")

# ===== USER'S COLLATE FUNCTION (collate_fn) =====
def collate_fn(batch):
    # Filter out samples that failed to load (returned None)
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None, None, None # Return four None objects if the batch is empty

    # Unpack tuple including filepath
    waveforms, labels, filepaths = zip(*batch)
    waveforms = [w.numpy() for w in waveforms]

    # Process and pad audio samples using the Wav2Vec2 processor
    inputs = processor(
        waveforms,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    )
    labels = torch.tensor(labels)
    # Return filepaths along with tensors
    return inputs.input_values, inputs.attention_mask, labels, filepaths

# --- FINAL DATA LOADING FUNCTION ---
def get_dataloaders():
    train_dataset = DogVoiceDataset(TRAIN_DIR, is_test=False)
    test_dataset = DogVoiceDataset(TEST_DIR, is_test=True)

    print(f"Number of training samples found: {len(train_dataset)}")
    print(f"Number of testing samples found: {len(test_dataset)}")
    print(f"Test label map (index -> name): {test_dataset.index_to_name}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader, test_dataset.index_to_name

# --- MODEL TRAINING AND EVALUATION ---

def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs):
    print(f"\n🚀 Starting training...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for input_values, attention_mask, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            if input_values is None: continue # Skip empty batches
            
            input_values = input_values.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(input_values, attention_mask=attention_mask).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch}/{num_epochs} — Average Loss: {avg_loss:.4f}")

def evaluate_model(model, test_loader, index_to_name):
    print("\n🧪 Starting evaluation...")
    model.eval()
    correct_predictions = 0
    total_samples = 0
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    
    # Lists to store detailed logging for both correct and incorrect predictions
    correct_predictions_log = []
    misclassified_samples_log = []

    with torch.no_grad():
        for input_values, attention_mask, labels, filepaths in test_loader:
            if input_values is None: continue # Skip empty batches
            
            input_values = input_values.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(input_values, attention_mask=attention_mask).logits
            
            # --- Calculate Softmax Probabilities for Confidence ---
            probabilities = torch.softmax(outputs, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
            
            # Iterate through the batch to check individual samples
            for i in range(len(labels)):
                total_samples += 1
                label = labels[i].item()
                predicted = predictions[i].item()
                class_total[label] += 1
                
                actual_name = index_to_name.get(label, f"Unknown({label})")
                predicted_name = index_to_name.get(predicted, f"Unknown({predicted})")
                
                # Get the top 3 prediction indices and sort by probability
                top_3_values, top_3_indices = torch.topk(probabilities[i], 3)
                
                # Format confidence list for logging
                confidence_list = []
                for j in range(3):
                    conf_name = index_to_name.get(top_3_indices[j].item(), f"Unknown({top_3_indices[j].item()})")
                    confidence_list.append((conf_name, top_3_values[j].item()))

                log_entry = {
                    "actual": actual_name,
                    "predicted": predicted_name,
                    "filepath": filepaths[i],
                    "confidence": confidence_list
                }
                
                if predicted == label:
                    correct_predictions += 1
                    class_correct[label] += 1
                    correct_predictions_log.append(log_entry)
                else:
                    misclassified_samples_log.append(log_entry)

    # --- Output Results ---
    
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    print(f"\n🎯 Overall Test Accuracy: {accuracy * 100:.2f}% (Total Correct: {correct_predictions}/{total_samples})")

    print("\nPer-Class Accuracy:")
    for i in range(NUM_CLASSES):
        name = index_to_name.get(i, f"Unknown({i})")
        total = class_total[i]
        if total > 0:
            acc = class_correct[i] / total
            print(f"  - {name} (Label {i}): {acc * 100:.2f}% (Correct: {class_correct[i]}/{total})")
        else:
            print(f"  - {name} (Label {i}): No samples found in test set.")

    # --- New Section for Correct Predictions ---
    print("\n✅ Successful Translations (Top 3 Confidence Scores):")
    for sample in correct_predictions_log:
        # Confidence is already sorted top-to-bottom
        top_scores = " | ".join([f"{name}: {score:.2f}" for name, score in sample['confidence']])
        print(f"  - File: {os.path.basename(sample['filepath'])}")
        print(f"    Actual/Predicted: '{sample['actual']}'")
        print(f"    Top 3 Confidences: {top_scores}")

    # --- Existing Section for Missed Predictions ---
    print("\n❌ Missed Translations (Misclassifications) with Top 3 Confidence:")
    for sample in misclassified_samples_log:
        top_scores = " | ".join([f"{name}: {score:.2f}" for name, score in sample['confidence']])
        print(f"  - File: {os.path.basename(sample['filepath'])}")
        print(f"    Actual: '{sample['actual']}' was translated as: '{sample['predicted']}'")
        print(f"    Top 3 Confidences: {top_scores}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Load data
    train_loader, test_loader, index_to_name = get_dataloaders()
    
    # Initialize model
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    
    # Apply class weights if provided (optimized using HPO)
    if CLASS_WEIGHTS is not None:
        # FIX: CLASS_WEIGHTS is already a tensor, so we just move it to the device.
        weights_tensor = CLASS_WEIGHTS.to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        print("💡 Using custom class weights for loss function.")
    else:
        # Standard CrossEntropyLoss if no weights are specified
        criterion = nn.CrossEntropyLoss()
        
    model.to(DEVICE)
    
    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # Check if train_loader has any data before calculating total_steps
    if len(train_loader) > 0:
        total_steps = len(train_loader) * N_EPOCHS
        scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    else:
        print("Warning: Train loader is empty. Skipping scheduler initialization.")
        # Create a dummy scheduler or handle this case
        total_steps = 1 # Prevent division by zero
        scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=total_steps)


    # Train and Evaluate
    train_model(model, train_loader, criterion, optimizer, scheduler, N_EPOCHS)
    evaluate_model(model, test_loader, index_to_name)
    
    # Save Model (omitted for short run)
    # torch.save(model.state_dict(), 'dog_translator_model.pth')
    # print("\n💾 Model saved as 'dog_translator_model.pth'")