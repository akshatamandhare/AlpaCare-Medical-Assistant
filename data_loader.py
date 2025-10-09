
import os
import json
import pandas as pd
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import random

class MedInstructDataLoader:
    def __init__(self, dataset_name="lavita/AlpaCare-MedInstruct-52k", subset_size=None):
        """
        Initialize the data loader for AlpaCare MedInstruct dataset
        
        Args:
            dataset_name: HuggingFace dataset identifier
            subset_size: Number of samples to use (None for all data)
        """
        self.dataset_name = dataset_name
        self.subset_size = subset_size
        self.tokenizer = None
        
    def load_dataset(self):
        """Load the dataset from HuggingFace"""
        try:
            print(f"Loading dataset: {self.dataset_name}")
            dataset = load_dataset(self.dataset_name)
            print(f"Dataset loaded successfully. Total samples: {len(dataset['train'])}")
            return dataset
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def clean_and_filter_data(self, dataset):
        """
        Clean and filter the dataset for safety and quality
        """
        def is_safe_content(example):
            """Check if content is safe (no diagnosis, prescription, etc.)"""
            unsafe_keywords = [
                'diagnose', 'prescribe', 'dosage', 'medication schedule',
                'treatment plan', 'clinical decision', 'medical emergency'
            ]
            
            text_content = f"{example['instruction']} {example.get('input', '')} {example['output']}"
            text_lower = text_content.lower()
            
            # Check for unsafe keywords
            for keyword in unsafe_keywords:
                if keyword in text_lower:
                    return False
            return True
        
        def add_disclaimer(example):
            """Add medical disclaimer to output"""
            disclaimer = "\\n\\n**Medical Disclaimer:** This information is for educational purposes only and should not be used for diagnosis, treatment, or medical decision-making. Always consult with a qualified healthcare professional for medical advice."
            
            example['output'] = example['output'] + disclaimer
            return example
        
        # Filter unsafe content
        print("Filtering unsafe content...")
        safe_dataset = dataset.filter(is_safe_content)
        print(f"Samples after safety filtering: {len(safe_dataset['train'])}")
        
        # Add disclaimers
        print("Adding medical disclaimers...")
        safe_dataset = safe_dataset.map(add_disclaimer)
        
        return safe_dataset
    
    def create_prompt_format(self, example):
        """
        Format examples into instruction-following format
        """
        if example.get('input', '').strip() == '' or example.get('input', '').strip() == '<noinput>':
            prompt = f"### Instruction:\\n{example['instruction']}\\n\\n### Response:\\n{example['output']}"
        else:
            prompt = f"### Instruction:\\n{example['instruction']}\\n\\n### Input:\\n{example['input']}\\n\\n### Response:\\n{example['output']}"
        
        return {"text": prompt}
    
    def tokenize_function(self, examples):
        """Tokenize the formatted examples"""
        return self.tokenizer(examples["text"], truncation=True, padding=False, max_length=512)
    
    def create_train_val_split(self, dataset, train_ratio=0.9, val_ratio=0.05, test_ratio=0.05):
        """
        Split dataset into train/validation/test sets
        """
        # Use only train split from the original dataset
        full_dataset = dataset['train']
        
        # Shuffle the dataset
        full_dataset = full_dataset.shuffle(seed=42)
        
        # If subset_size is specified, take only that many samples
        if self.subset_size:
            full_dataset = full_dataset.select(range(min(self.subset_size, len(full_dataset))))
            print(f"Using subset of {len(full_dataset)} samples")
        
        # Calculate split sizes
        total_size = len(full_dataset)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        # Create splits
        train_dataset = full_dataset.select(range(0, train_size))
        val_dataset = full_dataset.select(range(train_size, train_size + val_size))
        test_dataset = full_dataset.select(range(train_size + val_size, total_size))
        
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
    
    def preprocess_dataset(self, tokenizer_name="meta-llama/Llama-2-7b-hf"):
        """
        Complete preprocessing pipeline
        """
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load dataset
        dataset = self.load_dataset()
        if dataset is None:
            return None
        
        # Clean and filter
        dataset = self.clean_and_filter_data(dataset)
        
        # Create train/val/test splits
        split_dataset = self.create_train_val_split(dataset)
        
        # Format prompts
        formatted_dataset = split_dataset.map(self.create_prompt_format, remove_columns=split_dataset['train'].column_names)
        
        print(f"Final dataset sizes:")
        print(f"  Train: {len(formatted_dataset['train'])}")
        print(f"  Validation: {len(formatted_dataset['validation'])}")
        print(f"  Test: {len(formatted_dataset['test'])}")
        
        return formatted_dataset
    
    def save_dataset_info(self, dataset, output_dir="./"):
        """Save dataset information and sample data"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save dataset statistics
        stats = {
            "dataset_name": self.dataset_name,
            "total_samples": len(dataset['train']) + len(dataset['validation']) + len(dataset['test']),
            "train_samples": len(dataset['train']),
            "validation_samples": len(dataset['validation']),
            "test_samples": len(dataset['test']),
            "subset_size": self.subset_size
        }
        
        with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
            json.dump(stats, f, indent=2)
        
        # Save sample data
        sample_data = []
        for i in range(min(5, len(dataset['train']))):
            sample_data.append(dataset['train'][i])
        
        with open(os.path.join(output_dir, "sample_data.json"), "w") as f:
            json.dump(sample_data, f, indent=2)
        
        print(f"Dataset info saved to {output_dir}")


def main():
    # Example usage
    print("AlpaCare Medical Assistant Data Loader")
    print("=" * 40)
    
    # Initialize data loader with subset for faster processing
    loader = MedInstructDataLoader(subset_size=1000)  # Use 1000 samples for demo
    
    # Preprocess dataset
    dataset = loader.preprocess_dataset()
    
    if dataset:
        # Save dataset information
        loader.save_dataset_info(dataset)
        
        # Print a sample
        print("\\nSample formatted data:")
        print(dataset['train'][0]['text'][:500] + "...")
        
        return dataset
    else:
        print("Failed to load and preprocess dataset")
        return None


if __name__ == "__main__":

    main()
