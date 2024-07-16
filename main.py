import pandas as pd

# Load the CSV file
file_path = 'data truncated.csv'

# Try reading the CSV file with a different encoding
data = pd.read_csv(file_path, encoding='latin1')

# Display the first few rows of the dataframe to understand its structure
data.head()

# Extract the "Content" column
content_data = data['Content'].dropna().tolist()

# Display the first few rows to verify
print(content_data[:5])


from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Load and prepare the dataset
def load_dataset(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    conversations = []
    for line in lines:
        if line.strip():
            conversations.append(line.strip())

    return conversations

conversations = load_dataset('chat_data.txt')

# Tokenize the dataset
inputs = tokenizer(conversations, return_tensors='pt', padding=True, truncation=True)

# Move tensors to GPU if available
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)
#inputs = {key: val.to(device) for key, val in inputs.items()}

from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm

class ChatDataset(Dataset):
    def __init__(self, conversations, tokenizer):
        self.conversations = conversations
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.conversations[idx], return_tensors='pt')
        return inputs.input_ids.squeeze(), inputs.attention_mask.squeeze()

chat_dataset = ChatDataset(cleaned_content_data, tokenizer)
train_loader = DataLoader(chat_dataset, batch_size=2, shuffle=True)

optimizer = AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(3):  # Number of epochs
    for batch in tqdm(train_loader):
        input_ids, attention_mask = batch
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
