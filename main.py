import os
import re
import tkinter as tk
from tkinter import filedialog
import unicodedata
import pandas as pd
import ast
import torch

root = tk.Tk()
root.withdraw()

LINES_FILE = filedialog.askopenfilename(
    title="Select lines file",
    filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
)
CONVERSATIONS_FILE = filedialog.askopenfilename(
    title="Select conversations file",
    filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
)

# Special tokens used for sequence-to-sequence models
PAD_TOKEN = '<pad>' # Used to pad sequences to a uniform length
SOS_TOKEN = '<sos>' # Start of Sentence token
EOS_TOKEN = '<eos>' # End of Sentence token
UNK_TOKEN = '<unk>' # Unknown word token (for words not in vocabulary)

# Maximum sequence length for padding/truncation.
# This value is critical and should be chosen based on your dataset's
# sentence length distribution and your model's capacity.
# Too short, and you lose information; too long, and training becomes slow.
MAX_LENGTH = 15 # Example: set to 15 words/tokens per sentence

# --- 1. Data Loading and Parsing with Pandas ---

def load_dataframes():
    """
    Loads movie lines and conversations into Pandas DataFrames from TSV files.
    Assumes TSV files are tab-separated and have no header row,
    with columns in a specific order.
    """
    print(f"Attempting to load lines from: {LINES_FILE}")
    print(f"Attempting to load conversations from: {CONVERSATIONS_FILE}")

    df_lines = None
    df_conversations = None

    try:
        # Load movie_lines.tsv: lineID, characterID, movieID, characterName, text
        df_lines = pd.read_csv(LINES_FILE, sep='\t', header=None,
                               names=['lineID', 'characterID', 'movieID', 'characterName', 'text'],
                               encoding='iso-8859-1')
        print(f"Successfully loaded {len(df_lines)} lines.")
    except FileNotFoundError:
        print(f"Error: {LINES_FILE} not found. Please check DATA_PATH and file name.")
        return None, None
    except Exception as e:
        print(f"Error loading movie_lines.tsv: {e}")
        print("Please ensure the file is tab-separated with 5 columns and no header.")
        return None, None

    try:
        # Load movie_conversations.tsv: characterID1, characterID2, movieID, lineIDs_list
        df_conversations = pd.read_csv(CONVERSATIONS_FILE, sep='\t', header=None,
                                       names=['characterID1', 'characterID2', 'movieID', 'lineIDs_list'],
                                       encoding='iso-8859-1')
        # Convert the 'lineIDs_list' column (which is a string representation of a list)
        # into actual Python lists using ast.literal_eval for safe parsing.
        df_conversations['lineIDs_list'] = df_conversations['lineIDs_list'].apply(ast.literal_eval)
        print(f"Successfully loaded {len(df_conversations)} conversations.")
    except FileNotFoundError:
        print(f"Error: {CONVERSATIONS_FILE} not found. Please check DATA_PATH and file name.")
        return None, None
    except Exception as e:
        print(f"Error loading movie_conversations.tsv: {e}")
        print("Please ensure the file is tab-separated with 4 columns and no header.")
        return None, None

    return df_lines, df_conversations

def extract_qa_pairs_from_df(df_lines, df_conversations):
    """
    Extracts question-answer (input-response) pairs from the loaded DataFrames.
    Each pair represents a turn in a conversation.
    """
    qa_pairs = []
    if df_lines is None or df_conversations is None:
        print("Cannot extract QA pairs: DataFrames not loaded.")
        return qa_pairs

    # Create a dictionary for fast lookup of line text by its ID
    line_id_to_text = df_lines.set_index('lineID')['text'].to_dict()

    # Iterate through each conversation in the conversations DataFrame
    for index, row in df_conversations.iterrows():
        conv_ids = row['lineIDs_list'] # Get the list of line IDs for the current conversation
        # Create (input, response) pairs from consecutive lines in the conversation
        for i in range(len(conv_ids) - 1):
            input_line_id = conv_ids[i]
            response_line_id = conv_ids[i+1]

            # Retrieve the actual text for the line IDs
            input_line = line_id_to_text.get(input_line_id)
            response_line = line_id_to_text.get(response_line_id)

            # Only add the pair if both input and response lines were found
            if input_line and response_line:
                qa_pairs.append((input_line, response_line))
    return qa_pairs

# --- 2. Text Preprocessing Functions ---

def unicode_to_ascii(s):
    """
    Converts a unicode string to plain ASCII, removing accents and diacritics.
    This helps in normalizing text for consistent processing.
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def clean_text(text):
    """
    Cleans a given text string by:
    1. Converting to ASCII and lowercasing.
    2. Adding spaces around punctuation for easier tokenization.
    3. Replacing multiple spaces with a single space.
    4. Removing characters that are not letters, punctuation, or spaces.
    5. Stripping leading/trailing whitespace.
    """
    if not isinstance(text, str):
        return "" # Return empty string for non-string inputs

    text = unicode_to_ascii(text.lower().strip())
    # Add a space before and after common punctuation marks
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    # Replace multiple spaces with a single space
    text = re.sub(r'[" "]+', " ", text)
    # Remove any characters that are not a-z, A-Z, or the specified punctuation/space
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
    text = text.strip() # Remove leading/trailing whitespace
    return text

# --- 3. Vocabulary Creation Class ---

class Vocabulary:
    """
    Manages the mapping between words and their numerical indices.
    Includes special tokens for padding, start/end of sentence, and unknown words.
    """
    def __init__(self, name="corpus"):
        self.name = name
        self.word2index = {} # Maps word string to integer ID
        self.word2count = {} # Counts frequency of each word
        # Initialize with special tokens
        self.index2word = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN, 3: UNK_TOKEN}
        self.num_words = 4 # Current count of unique words including special tokens

    def add_sentence(self, sentence):
        """Adds all words in a sentence to the vocabulary."""
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        """Adds a single word to the vocabulary or increments its count."""
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

# --- 4. Tokenization Function ---

def indexes_from_sentence(vocab, sentence):
    """
    Converts a sentence (string) into a list of word indices.
    Words not in the vocabulary are mapped to UNK_TOKEN.
    An EOS_TOKEN is appended to the end of the sequence.
    """
    # Get index for each word, defaulting to UNK_TOKEN if word not found
    indexed_words = [vocab.word2index.get(word, vocab.word2index[UNK_TOKEN]) for word in sentence.split(' ')]
    # Append EOS_TOKEN to mark the end of the sentence
    return indexed_words + [vocab.word2index[EOS_TOKEN]]

# --- 5. Padding and Batching Functions ---

def pad_sequence(sequence, max_len, pad_token_id):
    """
    Pads a numerical sequence to a specified maximum length.
    If the sequence is longer than max_len, it is truncated.
    Shorter sequences are padded with `pad_token_id`.
    """
    # Truncate if longer than max_len
    truncated_seq = sequence[:max_len]
    # Pad if shorter than max_len
    padded_seq = truncated_seq + [pad_token_id] * (max_len - len(truncated_seq))
    return padded_seq

def prepare_batch(qa_batch, vocab, max_length):
    """
    Prepares a batch of (input, response) pairs for model training.
    This involves:
    1. Tokenizing sentences into numerical indices.
    2. Padding/truncating sequences to MAX_LENGTH.
    3. Converting lists of indices to PyTorch LongTensors.
    4. Calculating original (unpadded) lengths for potential use in models
       (e.g., for `pack_padded_sequence` in RNNs).
    """
    input_batch_indices = []
    target_batch_indices = []
    input_lengths = []
    target_lengths = []

    # Get the ID for the padding token from the vocabulary
    pad_id = vocab.word2index[PAD_TOKEN]

    for q, a in qa_batch:
        # Convert sentences to lists of word indices
        input_ids = indexes_from_sentence(vocab, q)
        target_ids = indexes_from_sentence(vocab, a)

        # Pad/truncate sequences
        input_batch_indices.append(pad_sequence(input_ids, max_length, pad_id))
        target_batch_indices.append(pad_sequence(target_ids, max_length, pad_id))

        # Store the original lengths (before padding/truncation)
        # This is useful for packed sequences in RNNs
        input_lengths.append(min(len(input_ids), max_length))
        target_lengths.append(min(len(target_ids), max_length))

    # Convert lists of lists to PyTorch LongTensors
    # LongTensors are suitable for storing integer indices (like word IDs)
    input_tensor = torch.LongTensor(input_batch_indices)
    target_tensor = torch.LongTensor(target_batch_indices)

    return input_tensor, target_tensor, input_lengths, target_lengths


# --- Main Execution Flow ---
if __name__ == "__main__":
    print("--- Starting Data Preprocessing Pipeline ---")

    # Step 1: Load DataFrames
    df_lines, df_conversations = load_dataframes()

    # Only proceed if both DataFrames were loaded successfully
    if df_lines is not None and df_conversations is not None:
        # Step 2: Extract QA Pairs from DataFrames
        print("\nExtracting QA pairs from loaded DataFrames...")
        qa_data_list = extract_qa_pairs_from_df(df_lines, df_conversations)
        print(f"Initial raw QA pairs extracted: {len(qa_data_list)}")

        # Convert the list of tuples to a Pandas DataFrame for easier manipulation
        df_qa = pd.DataFrame(qa_data_list, columns=['input', 'response'])
        print("\nDataFrame head before cleaning:")
        print(df_qa.head())

        # Step 3: Apply Text Cleaning
        print("\nApplying text cleaning to 'input' and 'response' columns...")
        df_qa['input'] = df_qa['input'].apply(clean_text)
        df_qa['response'] = df_qa['response'].apply(clean_text)

        # Step 4: Post-cleaning Filtering
        # Remove rows where either input or response became empty after cleaning
        initial_rows = len(df_qa)
        df_qa.replace('', pd.NA, inplace=True) # Replace empty strings with Pandas' NA
        df_qa.dropna(subset=['input', 'response'], inplace=True) # Drop rows with NA in these columns
        df_qa.reset_index(drop=True, inplace=True) # Reset index after dropping rows

        print(f"Removed {initial_rows - len(df_qa)} rows that became empty after cleaning.")
        print(f"Final cleaned QA pairs in DataFrame: {len(df_qa)}")

        print("\nDataFrame head after cleaning and filtering:")
        print(df_qa.head())

        # Step 5: Build Vocabulary
        print("\nBuilding vocabulary...")
        vocab = Vocabulary("movie_dialogs")
        # Add all words from both input and response columns to the vocabulary
        for sentence in df_qa['input']:
            vocab.add_sentence(sentence)
        for sentence in df_qa['response']:
            vocab.add_sentence(sentence)

        print(f"Vocabulary built. Total unique words (including special tokens): {vocab.num_words}")
        print(f"Example words in vocabulary: {list(vocab.word2index.items())[:10]}")
        print(f"Index of '{SOS_TOKEN}': {vocab.word2index[SOS_TOKEN]}")
        print(f"Word at index 5: {vocab.index2word.get(5, 'Not Found')}")


        # Step 6: Demonstrate Tokenization and Batching
        print(f"\nDemonstrating tokenization and batching with MAX_LENGTH={MAX_LENGTH}...")
        sample_batch_size = 5
        if len(df_qa) >= sample_batch_size:
            # Take a small sample of cleaned QA pairs for demonstration
            sample_qa_batch = df_qa.sample(n=sample_batch_size).values.tolist() # Get as list of lists
            print(f"\nSample raw QA pairs for batching:")
            for q, a in sample_qa_batch:
                print(f"  Q: {q}\n  A: {a}")

            # Prepare the batch
            input_tensor, target_tensor, input_lengths, target_lengths = prepare_batch(
                sample_qa_batch, vocab, MAX_LENGTH
            )

            print(f"\nPrepared Batch Details (Batch Size: {sample_batch_size}, MAX_LENGTH: {MAX_LENGTH}):")
            print(f"Input Tensor Shape: {input_tensor.shape}")
            print(f"Target Tensor Shape: {target_tensor.shape}")
            print(f"Input Lengths (original, before padding): {input_lengths}")
            print(f"Target Lengths (original, before padding): {target_lengths}")

            print("\nExample Input Tensor (first sample):")
            print(input_tensor[0])
            print("Example Target Tensor (first sample):")
            print(target_tensor[0])

            # Convert back to words for human readability (first sample)
            print("\nFirst sample converted back to words:")
            input_words = [vocab.index2word.get(idx.item(), UNK_TOKEN) for idx in input_tensor[0]]
            target_words = [vocab.index2word.get(idx.item(), UNK_TOKEN) for idx in target_tensor[0]]
            print(f"Input (words): {' '.join(input_words)}")
            print(f"Response (words): {' '.join(target_words)}")

        else:
            print("Not enough data to create a sample batch for demonstration.")

    print("\n--- Data Preprocessing Pipeline Complete ---")