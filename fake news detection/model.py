import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
import pandas as pd


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)


def tokenize_data(texts):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=512
    )

def prepare_data(df):
    tokenized_data = tokenize_data(df['cleaned_text'].tolist())
    labels = torch.tensor(df['label'].values)
    dataset = TensorDataset(tokenized_data['input_ids'], tokenized_data['attention_mask'], labels)
    return DataLoader(dataset, batch_size=8, shuffle=True)


def train_model(train_dataloader):
    optimizer = AdamW(model.parameters(), lr=1e-5)
    model.train()

    for epoch in range(3):
        total_loss = 0
        for batch in train_dataloader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} - Loss: {total_loss / len(train_dataloader):.4f}")


def save_model():
    model.save_pretrained("./model")
    tokenizer.save_pretrained("./model")


if __name__ == "__main__":

    df_fake = pd.read_csv("fake news detection/Fake.csv")
    df_true = pd.read_csv("fake news detection/True.csv")

    
    df_fake['label'] = 0
    df_true['label'] = 1

    
    df = pd.concat([df_fake, df_true]).sample(frac=1).reset_index(drop=True)

   
    df['cleaned_text'] = df['text']

   
    train_loader = prepare_data(df)
    train_model(train_loader)
    save_model()
