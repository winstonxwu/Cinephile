import os
import re

DATA_PATH = '/Users/winstonxwu/Downloads/archive'
LINES_FILE = os.path.join(DATA_PATH, 'movie_lines.txt')
CONVERSATIONS_FILE = os.path.join(DATA_PATH, 'movie_conversations.txt')

def load_lines():
    lines = {}
    with open(LINES_FILE, 'r', encoding='iso-8859-1') as f:
        for line in f:
            parts = line.split(' +++$+++ ')
            if len(parts) == 5:
                line_id = parts[0]
                text = parts[4].strip()
                lines[line_id] = text
    return lines

def load_conversations():
    conversations = []
    with open(CONVERSATIONS_FILE, 'r', encoding='iso-8859-1') as f:
        conversations = []
        for line in f:
            parts = line.split(' +++$+++ ')
            if len(parts) == 4:
                line_ids_str = parts[3].strip()
                try:
                    line_ids = eval(line_ids_str)
                    conversations.append(line_ids)
                except SyntaxError:
                    print(f"Skipping malformed line ID String: {line_ids_str}")
        return conversations
    
def extract_qa_pairs(lines, conversations):
    qa_pairs = []
    for conv_ids in conversations:
        for i in range(len(conv_ids)-1):
            input_line = lines.get(conv_ids[i])
            response_line = lines.get(conv_ids[i+1])
            if input_line and response_line:
                qa_pairs.append((input_line, response_line))
    return qa_pairs

print("Loading lines ...")
lines_dict = load_lines()
print(f"Loaded {len(lines_dict)} lines.")

print("Loading conversations...")
conversations_list = load_conversations()
print(f"Loaded {len(conversations_list)} conversations.")

print("Extracting QA pairs...")
qa_data = extract_qa_pairs(lines_dict, conversations_list)
print(f"Extracted {len(qa_data)} QA pairs.")

if qa_data:
    print("\nExample QA pair:")
    print(f"Input: {qa_data[0][0]}")
    print(f"Response: {qa_data[0][1]}")

import unicodedata

def unicode_to_ascii(s):
    return ''.koin(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def clean_text(text):
    text = unicode_to_ascii(text.lower().strip())
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
    text = text.strip()
    return text

cleaned_qa_data = []
for q, a in qa_data:
    cleaned_q = clean_text(q)
    cleaned_a = clean_text(a)
    # You might want to filter out empty strings after cleaning
    if cleaned_q and cleaned_a:
        cleaned_qa_data.append((cleaned_q, cleaned_a))

print(f"\nCleaned {len(cleaned_qa_data)} QA pairs.")
if cleaned_qa_data:
    print("\nExample Cleaned QA pair:")
    print(f"Input: {cleaned_qa_data[0][0]}")
    print(f"Response: {cleaned_qa_data[0][1]}")

PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>' # Start of Sentence
EOS_TOKEN = '<eos>' # End of Sentence
UNK_TOKEN = '<unk>' # Unknown word

class Vocabulary:
    def __init__(self, name="corpus"):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN, 3: UNK_TOKEN}
        self.num_words = 4 # Count SOS, EOS, PAD, UNK

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

# Build vocabulary
vocab = Vocabulary("movie_dialogs")
for q, a in cleaned_qa_data:
    vocab.add_sentence(q)
    vocab.add_sentence(a)

print(f"\nVocabulary size: {vocab.num_words} words.")
# print(list(vocab.word2index.items())[:10]) # Example words

# Convert sentences to sequences of word IDs
def indexes_from_sentence(vocab, sentence):
    return [vocab.word2index.get(word, vocab.word2index[UNK_TOKEN]) for word in sentence.split(' ')] + [vocab.word2index[EOS_TOKEN]]

# Example
indexed_q = indexes_from_sentence(vocab, cleaned_qa_data[0][0])
indexed_a = indexes_from_sentence(vocab, cleaned_qa_data[0][1])
print(f"\nExample indexed input: {indexed_q}")
print(f"Example indexed response: {indexed_a}")

MAX_LENGTH = 10 # Define a reasonable max length for your sequences

def pad_sequence(sequence, max_len, pad_token_id):
    """Pads a sequence to max_len with pad_token_id."""
    padded_seq = sequence[:max_len] + [pad_token_id] * (max_len - len(sequence))
    return padded_seq

def prepare_batch(qa_batch, vocab, max_length):
    """
    Converts a list of (query, response) strings into padded numerical tensors.
    """
    input_batch = []
    target_batch = []
    input_lengths = []
    target_lengths = []

    for q, a in qa_batch:
        input_ids = indexes_from_sentence(vocab, q)
        target_ids = indexes_from_sentence(vocab, a)

        input_batch.append(pad_sequence(input_ids, max_length, vocab.word2index[PAD_TOKEN]))
        target_batch.append(pad_sequence(target_ids, max_length, vocab.word2index[PAD_TOKEN]))

        input_lengths.append(min(len(input_ids), max_length))
        target_lengths.append(min(len(target_ids), max_length))

    # Convert to PyTorch tensors (if using PyTorch)
    import torch
    input_tensor = torch.LongTensor(input_batch)
    target_tensor = torch.LongTensor(target_batch)

    return input_tensor, target_tensor, input_lengths, target_lengths

# Example of preparing a batch
batch_size = 5
example_batch = cleaned_qa_data[:batch_size]

input_tensor, target_tensor, input_lengths, target_lengths = prepare_batch(
    example_batch, vocab, MAX_LENGTH
)

print(f"\nExample input tensor shape: {input_tensor.shape}") # Should be [batch_size, MAX_LENGTH]
print(f"Example target tensor shape: {target_tensor.shape}")
print(f"Example input tensor:\n{input_tensor}")
print(f"Example target tensor:\n{target_tensor}")