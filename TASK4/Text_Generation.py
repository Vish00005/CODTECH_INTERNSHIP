"""
Text Generation Model using DistilGPT-2

This script demonstrates how to fine-tune a pre-trained DistilGPT-2 model on a text dataset
and generate contextually relevant paragraphs based on user-provided prompts.

Libraries Required: numpy, pandas, matplotlib, tensorflow/keras or torch, transformers, datasets, nltk, scikit-learn
"""

import os
# Disable parallelism to avoid warnings and potential hangs in notebooks/scripts
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def main():
    print("="*50)
    print("1. Loading Pre-trained Model and Tokenizer")
    print("="*50)
    model_name = "MBZUAI/LaMini-GPT-124M" # Using an instruction-tuned GPT-2 for accurate, non-hallucinated explanations
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # GPT-2/Neo does not have a padding token by default, we assign the EOS token as the PAD token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load Model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print(f"Loaded {model_name} successfully.")

    print("\n" + "="*50)
    print("2. Data Collection & Preprocessing")
    print("="*50)
    # We use the Wikitext dataset for a broader vocabulary (Wikipedia articles) 
    # instead of news articles to prevent the model from generating random news blurbs.
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:500]")
    
    # Filter out empty or extremely short lines which are common in wikitext
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 10)
    
    print(f"Dataset loaded. Number of records: {len(dataset)}")
    print("Sample text:", dataset[0]["text"])

    # Tokenization and Text Cleaning Function
    def tokenize_function(examples):
        # The tokenizer handles lowercasing, punctuation, and sub-word encoding (BPE)
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=128
        )

    # Apply tokenization
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # For Causal Language Modeling, labels are the same as input_ids
    def copy_labels(example):
        example["labels"] = example["input_ids"].copy()
        return example

    tokenized_datasets = tokenized_datasets.map(copy_labels)
    
    # Vocabulary creation equivalent (word_to_index mapping handled by BPE)
    vocab = tokenizer.get_vocab()
    print(f"\nVocabulary size: {len(vocab)}")
    print("Sample tokens:", list(vocab.items())[:5])

    print("\n" + "="*50)
    print("3. Model Architecture & Training")
    print("="*50)
    
    print(model) # Display model architecture summary

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./gpt_results",
        num_train_epochs=1,
        per_device_train_batch_size=4, # Small batch size to fit in memory
        logging_steps=5,
        max_steps=20, # Very small number of steps to prevent catastrophic forgetting of pre-trained knowledge
        learning_rate=2e-5, # Low learning rate to gently fine-tune
        report_to="none"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    print("Starting fine-tuning...")
    trainer.train()
    print("Training complete.")

    # Visualization of Training Loss
    log_history = trainer.state.log_history
    loss_values = [log["loss"] for log in log_history if "loss" in log]
    steps = [log["step"] for log in log_history if "loss" in log]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, loss_values, marker='o', linestyle='-', color='b', label="Training Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Model Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("Training_Loss.png")
    print("Saved training loss plot as 'Training_Loss.png'")

    print("\n" + "="*50)
    print("4. Text Generation")
    print("="*50)
    
    def generate_text(prompt, max_new_tokens=100, temperature=0.7):
        """
        Generates continuation text based on the provided prompt.
        Uses Top-P sampling and Temperature scaling for varied, creative outputs.
        """
        # Prefix the prompt to force a descriptive continuation
        formatted_prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWrite a detailed paragraph about {prompt}.\n\n### Response:"
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        inputs = inputs.to(device)
        model.to(device)
        
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            min_new_tokens=20, # Force model to generate at least 20 new tokens
            temperature=temperature,
            top_p=0.9,       # Nucleus sampling
            do_sample=True,  # Enable sampling to use temperature/top_p
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2 # Reduce repeating phrases
        )
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Parse out the response portion
        return full_text.split("### Response:")[1].strip() if "### Response:" in full_text else full_text

    # Test Prompts
    prompts = [
        "Artificial Intelligence",
        "Climate Change",
        "Space Exploration",
        "Cyber Security",
        "Education Technology",
        "JEE Mains"
    ]

    print("--- Automatically Generated Prompts ---")
    for p in prompts:
        print(f"\n[Prompt]: {p}")
        generated = generate_text(p, max_new_tokens=60)
        print(f"[Generated Output]:\n{generated}")
        print("-" * 60)

    print("\n--- Manual Prompt Entry ---")
    while True:
        try:
            user_input = input("\nEnter your prompt (or type 'exit' to quit): ")
            if user_input.strip().lower() == 'exit':
                print("Exiting...")
                break
            if not user_input.strip():
                continue
            
            print(f"\n[Generating...]")
            generated = generate_text(user_input.strip(), max_new_tokens=60)
            print(f"[Generated Output]:\n{generated}")
            print("-" * 60)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except EOFError:
            break

if __name__ == "__main__":
    main()
