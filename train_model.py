import os
import torch
import wandb
import json
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from trl import SFTTrainer
from data_loader import MedInstructDataLoader


class AlpaCareTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.dataset = None
        
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with quantization"""
        print(f"Setting up model: {self.config['model_name']}")

        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model_name'],
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

        print("✅ Model and tokenizer loaded successfully!")

    def setup_lora(self):
        """Setup LoRA configuration"""
        print("Setting up LoRA...")

        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=self.config['lora']['lora_dropout'],
            bias=self.config['lora']['bias'],
            task_type=self.config['lora']['task_type'],
        )

        # Apply LoRA to the model
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        print("✅ LoRA configuration applied!")
        return lora_config
    
    def prepare_dataset(self):
        """Load and prepare the dataset"""
        print("Preparing dataset...")

        # Initialize data loader
        loader = MedInstructDataLoader(
            dataset_name=self.config['dataset_name'],
            subset_size=self.config.get('subset_size', None)
        )

        # Preprocess dataset
        self.dataset = loader.preprocess_dataset(self.config['model_name'])

        if self.dataset is None:
            raise Exception("Failed to load dataset")

        print("✅ Dataset prepared successfully!")
        return self.dataset
    
    def train_model(self):
        """Execute the training process"""
        print("Starting training...")
        
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Setup LoRA
        lora_config = self.setup_lora()
        
        # Prepare dataset
        dataset = self.prepare_dataset()
        
        # Training arguments
        training_args = TrainingArguments(**self.config['training'])
        
        # Create trainer
        # trainer = SFTTrainer(
        #     model=self.model,
        #     train_dataset=dataset['train'],
        #     eval_dataset=dataset['validation'],
        #     peft_config=lora_config,
        #     dataset_text_field="text",
        #     tokenizer=self.tokenizer,
        #     args=training_args,
        #     max_seq_length=512,
        #     packing=False,
        # )
        from trl import SFTTrainer, SFTConfig

        from trl import SFTTrainer, SFTConfig

        # Build the SFT configuration (this replaces TrainingArguments for SFTTrainer)
        sft_config = SFTConfig(
            output_dir=self.config["training"]["output_dir"],
            num_train_epochs=self.config["training"]["num_train_epochs"],
            per_device_train_batch_size=self.config["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
            per_device_eval_batch_size=self.config["training"]["per_device_eval_batch_size"],
            learning_rate=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
            logging_steps=self.config["training"]["logging_steps"],
            save_steps=self.config["training"]["save_steps"],
            max_grad_norm=self.config["training"]["max_grad_norm"],
            dataset_text_field="text",
            fp16=self.config["training"]["fp16"],
            warmup_ratio=self.config["training"]["warmup_ratio"],
            lr_scheduler_type=self.config["training"]["lr_scheduler_type"],
            packing=False,
            report_to=None,
        )


        # Create the trainer (no TrainingArguments, no output_dir here)
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            peft_config=lora_config,
            args=sft_config,   # pass SFTConfig here
        )




        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Start training
        start_time = datetime.now()
        trainer.train()
        end_time = datetime.now()
        
        training_time = end_time - start_time
        print(f"\n✅ Training completed in: {training_time}")
        
        # Save the model
        output_dir = self.config['training']['output_dir']
        print(f"Saving model to: {output_dir}")
        trainer.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training info
        self.save_training_info(training_time, output_dir)
        
        return trainer, output_dir
    
    def save_training_info(self, training_time, output_dir):
        """Save training configuration and info"""
        training_info = {
            "model_name": self.config['model_name'],
            "dataset_name": self.config['dataset_name'],
            "subset_size": self.config.get('subset_size'),
            "training_samples": len(self.dataset['train']) if self.dataset else 0,
            "config": self.config,
            "training_time": str(training_time),
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(os.path.join(output_dir, "training_info.json"), "w") as f:
            json.dump(training_info, f, indent=2)
        
        print("✅ Training information saved!")


def get_default_config():
    """Get default training configuration"""
    return {
        "model_name": "distilgpt2",  # smaller causal LM model
        "dataset_name": "lavita/AlpaCare-MedInstruct-52k",
        "subset_size": 5000,  # Use subset for faster training
        
        "lora": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": [
                "c_attn", "c_proj"
            ],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        },
        
        # "training": {
        #     "output_dir": "./alpacare-lora-adapter",
        #     "num_train_epochs": 1,
        #     "per_device_train_batch_size": 1,
        #     "gradient_accumulation_steps": 4,
        #     "per_device_eval_batch_size": 2,
        #     "optim": "adamw_torch",
        #     "save_steps": 100,
        #     "logging_steps": 10,
        #     "learning_rate": 2e-4,
        #     "weight_decay": 0.001,
        #     "fp16": True,
        #     "max_grad_norm": 0.3,
        #     "max_steps": 500,
        #     "warmup_ratio": 0.03,
        #     "group_by_length": True,
        #     "lr_scheduler_type": "constant",
        #     "evaluation_strategy": "steps",
        #     "eval_steps": 100,
        #     "save_total_limit": 2,
        #     "load_best_model_at_end": True,
        #     "report_to": None,  # Disable wandb for now
        # }
        "training": {
            "output_dir": "./alpacare-lora-adapter",
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "per_device_eval_batch_size": 2,
            "optim": "adamw_torch",
            "save_steps": 100,
            "logging_steps": 10,
            "learning_rate": 2e-4,
            "weight_decay": 0.001,
            "fp16": True,
            "max_grad_norm": 0.3,
            "max_steps": 500,
            "warmup_ratio": 0.03,
            "group_by_length": True,
            "lr_scheduler_type": "constant",
            "report_to": None,
        }

    }


def main():
    """Main training function"""
    print("🏥 AlpaCare Medical Assistant - Training Pipeline")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Training will be very slow on CPU.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    else:
        print(f"✅ Using GPU: {torch.cuda.get_device_name()}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Get configuration
    config = get_default_config()
    
    # Print configuration
    print("\n📋 Training Configuration:")
    print(f"  Model: {config['model_name']}")
    print(f"  Dataset: {config['dataset_name']}")
    print(f"  Subset Size: {config['subset_size']}")
    print(f"  Epochs: {config['training']['num_train_epochs']}")
    print(f"  Batch Size: {config['training']['per_device_train_batch_size']}")
    print(f"  Learning Rate: {config['training']['learning_rate']}")
    print(f"  Output Dir: {config['training']['output_dir']}")
    
    # Confirm training
    response = input("\n🚀 Start training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    try:
        # Initialize trainer
        trainer = AlpaCareTrainer(config)
        
        # Start training
        trained_model, output_dir = trainer.train_model()
        
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"✅ LoRA adapter saved to: {output_dir}")
        print(f"✅ You can now use the inference script or notebook")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

