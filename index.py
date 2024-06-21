import random
from entity import extract_entities 
from context import context_manager
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import json

# Example dataset
data = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "How are you", "Good morning", "Good evening"],
            "responses": ["Hello!", "Hi there!", "How can I help you?"]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "See you later", "Goodbye"],
            "responses": ["Goodbye!", "See you later!", "Have a nice day!"]
        },
        {
            "tag": "thanks",
            "patterns": ["Thanks", "Thank you", "That's helpful"],
            "responses": ["You're welcome!", "Happy to help!", "Anytime!"]
        },
        {
            "tag": "weather",
            "patterns": ["What's the weather like", "Is it going to rain today?", "Weather forecast"],
            "responses": ["Let me check the weather for you.", "Here's the weather forecast..."]
        }
    ]
}

# Save the dataset to a JSON file
with open('intents.json', 'w') as f:
    json.dump(data, f)


# Load the intents dataset
with open('intents.json') as f:
    intents = json.load(f)

# Prepare the dataset for training
class IntentsDataset(Dataset):
    def __init__(self, tokenizer, intents, max_length=16):
        self.tokenizer = tokenizer
        self.intents = intents
        self.max_length = max_length
        self.labels = {intent['tag']: idx for idx, intent in enumerate(intents['intents'])}
        self.data = [(pattern, self.labels[intent['tag']]) for intent in intents['intents'] for pattern in intent['patterns']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = IntentsDataset(tokenizer, intents)

# Fine-tune the BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(intents['intents']))

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset
)

trainer.train()


def classify_intent(text):
    inputs = tokenizer.encode_plus(
        text,
        truncation=True,
        padding='max_length',
        max_length=16,
        return_tensors='pt'
    )
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    intent_idx = torch.argmax(probs).item()
    intent = list(intents['intents'])[intent_idx]
    return intent['tag']

def get_response(intent, user_id):
    intent_data = next(item for item in intents['intents'] if item["tag"] == intent)
    response = random.choice(intent_data['responses'])
    return response

def chatbot_response(text, user_id):
    intent = classify_intent(text)
    entities = extract_entities(text)
    context_manager.update_context(user_id, 'last_intent', intent)
    response = get_response(intent, user_id)
    return response, entities

# Example interaction
user_id = 'user123'
text = "Hello"
response, entities = chatbot_response(text, user_id)
print(f"Bot: {response}")
print(f"Entities: {entities}")

text = "What's the weather like in London?"
response, entities = chatbot_response(text, user_id)
print(f"Bot: {response}")
print(f"Entities: {entities}")
