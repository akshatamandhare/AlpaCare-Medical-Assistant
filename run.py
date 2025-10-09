"""
AlpaCare Medical Assistant - Quick Start Script
One-command training and inference
"""

import os
import sys
import argparse
from train_model import AlpaCareTrainer, get_default_config
from inference import AlpaCareInference, get_sample_queries

def quick_train():
    """Quick training with default settings"""
    print("🚀 Quick Training - AlpaCare Medical Assistant")
    print("=" * 50)
    
    config = get_default_config()
    config['subset_size'] = 1000  # Small subset for quick training
    config['training']['max_steps'] = 100  # Fewer steps
    
    print("Using quick training settings:")
    print(f"- Subset size: {config['subset_size']}")
    print(f"- Max steps: {config['training']['max_steps']}")
    
    trainer = AlpaCareTrainer(config)
    model, output_dir = trainer.train_model()
    
    print(f"\\n✅ Quick training completed! Adapter saved to: {output_dir}")
    return output_dir

def quick_inference(adapter_path):
    """Quick inference with sample queries"""
    print("\\n🧠 Quick Inference - Testing the model...")
    
    inferencer = AlpaCareInference(adapter_path=adapter_path)
    inferencer.load_model()
    
    # Run sample queries
    sample_queries = get_sample_queries()[:3]  # Just first 3 queries
    results = inferencer.batch_inference(sample_queries)
    
    # Display results
    for result in results:
        print("\\n" + "-" * 60)
        print(f"Q: {result['instruction']}")
        print(f"A: {result['response'][:200]}...")  # Truncate for display
    
    filename = inferencer.save_results(results)
    print(f"\\n✅ Inference completed! Results saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description="AlpaCare Medical Assistant")
    parser.add_argument("command", choices=["train", "inference", "quick", "setup"], 
                       help="Command to run")
    parser.add_argument("--adapter-path", default="./alpacare-lora-adapter",
                       help="Path to LoRA adapter")
    
    args = parser.parse_args()
    
    if args.command == "setup":
        from setup import main as setup_main
        setup_main()
        
    elif args.command == "train":
        from train_model import main as train_main
        train_main()
        
    elif args.command == "inference":
        from inference import main as inference_main
        inference_main()
        
    elif args.command == "quick":
        # Quick training and inference
        adapter_path = quick_train()
        quick_inference(adapter_path)
        
        print("\\n🎉 Quick start completed!")
        print("Now you can:")
        print("- Run full training: python run.py train")
        print("- Run interactive inference: python run.py inference")

if __name__ == "__main__":
    main()