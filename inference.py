"""
AlpaCare Medical Assistant - Inference Script
Load trained model and run inference with safety checks
"""

import os
import torch
import json
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel
import warnings
warnings.filterwarnings('ignore')

class AlpaCareInference:
    def __init__(self, base_model="distilgpt2", adapter_path="./alpacare-lora-adapter"):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self.medical_disclaimer = "\\n\\n**Medical Disclaimer:** This information is for educational purposes only and should not be used for diagnosis, treatment, or medical decision-making. Always consult with a qualified healthcare professional for medical advice."
        
    def load_model(self):
        """Load base model and LoRA adapter"""
        print(f"Loading base model: {self.base_model}")
        
        # Configure quantization for inference
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Load LoRA adapter if exists
        if os.path.exists(self.adapter_path):
            print(f"Loading LoRA adapter from: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            print("✅ Model loaded with LoRA adapter!")
        else:
            print("⚠️  LoRA adapter not found. Using base model only.")
            self.model = base_model
            
    def format_prompt(self, instruction, input_text=""):
        """Format the prompt for the model"""
        if input_text.strip():
            prompt = f"### Instruction:\\n{instruction}\\n\\n### Input:\\n{input_text}\\n\\n### Response:\\n"
        else:
            prompt = f"### Instruction:\\n{instruction}\\n\\n### Response:\\n"
        return prompt
    
    def generate_response(self, instruction, input_text="", max_length=300, temperature=0.7, top_p=0.9):
        """Generate a response from the model with medical disclaimer"""
        
        # Format prompt
        prompt = self.format_prompt(instruction, input_text)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        
        # Add medical disclaimer if not already present
        if "Medical Disclaimer" not in response and "medical disclaimer" not in response.lower():
            response += self.medical_disclaimer
        
        return response
    
    def batch_inference(self, queries):
        """Run inference on multiple queries"""
        results = []
        
        for i, query in enumerate(queries, 1):
            print(f"Processing query {i}/{len(queries)}...")
            
            response = self.generate_response(
                query.get('instruction', ''),
                query.get('input', '')
            )
            
            result = {
                "query_id": i,
                "instruction": query.get('instruction', ''),
                "input": query.get('input', ''),
                "response": response,
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
            
        return results
    
    def save_results(self, results, filename=None):
        """Save inference results to JSON file"""
        if filename is None:
            filename = f"inference_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_data = {
            "model": self.base_model,
            "adapter_path": self.adapter_path,
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(results),
            "results": results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Results saved to: {filename}")
        return filename
    
    def interactive_mode(self):
        """Interactive question-answering mode"""
        print("\\n🏥 AlpaCare Medical Assistant - Interactive Mode")
        print("=" * 60)
        print("⚠️  REMINDER: This is for educational purposes only!")
        print("Always consult healthcare professionals for medical advice.")
        print("=" * 60)
        print("Type 'quit' to exit\\n")
        
        while True:
            try:
                instruction = input("🔹 Enter your medical question: ").strip()
                
                if instruction.lower() in ['quit', 'exit', 'q']:
                    print("Thank you for using AlpaCare Medical Assistant!")
                    break
                
                if not instruction:
                    print("Please enter a question.")
                    continue
                
                input_text = input("🔹 Additional context (optional, press Enter to skip): ").strip()
                
                print("\\n⏳ Generating response...")
                response = self.generate_response(instruction, input_text)
                
                print("\\n" + "=" * 80)
                print("🤖 AI RESPONSE")
                print("=" * 80)
                print(response)
                print("=" * 80 + "\\n")
                
            except KeyboardInterrupt:
                print("\\n\\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue

def get_sample_queries():
    """Get sample medical queries for testing"""
    return [
        {
            "instruction": "Explain what diabetes is in simple terms.",
            "input": ""
        },
        {
            "instruction": "What are the common symptoms of hypertension?",
            "input": ""
        },
        {
            "instruction": "Describe the importance of regular exercise for heart health.",
            "input": ""
        },
        {
            "instruction": "Explain the difference between Type 1 and Type 2 diabetes.",
            "input": ""
        },
        {
            "instruction": "What lifestyle changes can help prevent heart disease?",
            "input": ""
        }
    ]

def main():
    """Main inference function"""
    print("🏥 AlpaCare Medical Assistant - Inference")
    print("=" * 50)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"✅ Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  Using CPU (inference will be slower)")
    
    # Get adapter path from user
    adapter_path = input("\\n📁 Enter LoRA adapter path (press Enter for default './alpacare-lora-adapter'): ").strip()
    if not adapter_path:
        adapter_path = "./alpacare-lora-adapter"
    
    # Initialize inference
    try:
        inferencer = AlpaCareInference(adapter_path=adapter_path)
        inferencer.load_model()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Choose mode
    print("\\n🔧 Choose mode:")
    print("1. Interactive mode (ask questions)")
    print("2. Batch processing (sample queries)")
    print("3. Custom batch file")
    
    choice = input("\\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        # Interactive mode
        inferencer.interactive_mode()
        
    elif choice == "2":
        # Batch processing with sample queries
        print("\\n🔄 Running batch inference on sample queries...")
        sample_queries = get_sample_queries()
        results = inferencer.batch_inference(sample_queries)
        
        # Display results
        for result in results:
            print("\\n" + "=" * 80)
            print(f"Query {result['query_id']}: {result['instruction']}")
            if result['input']:
                print(f"Input: {result['input']}")
            print("\\nResponse:")
            print(result['response'])
            print("=" * 80)
        
        # Save results
        filename = inferencer.save_results(results)
        print(f"\\n✅ Batch processing completed. Results saved to: {filename}")
        
    elif choice == "3":
        # Custom batch file
        batch_file = input("\\n📄 Enter path to JSON file with queries: ").strip()
        if os.path.exists(batch_file):
            try:
                with open(batch_file, 'r') as f:
                    queries = json.load(f)
                
                results = inferencer.batch_inference(queries)
                filename = inferencer.save_results(results)
                print(f"\\n✅ Custom batch processing completed. Results saved to: {filename}")
                
            except Exception as e:
                print(f"❌ Error processing batch file: {e}")
        else:
            print(f"❌ File not found: {batch_file}")
    
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()