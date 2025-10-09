"""
AlpaCare Medical Assistant - Setup Script
Install dependencies and verify environment
"""

import subprocess
import sys
import os
import torch

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_cuda():
    """Check CUDA availability"""
    print("\\n🔍 Checking CUDA availability...")
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.version.cuda}")
        print(f"✅ GPU: {torch.cuda.get_device_name()}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("⚠️  CUDA not available. Training will be slow on CPU.")
        return False

def install_requirements():
    """Install all required packages"""
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.35.0", 
        "peft>=0.6.0",
        "datasets>=2.14.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.41.0",
        "trl>=0.7.0",
        "scipy",
        "scikit-learn",
        "evaluate",
        "rouge-score",
        "jupyter",
        "ipywidgets"
    ]
    
    print("📦 Installing requirements...")
    failed_packages = []
    
    for package in requirements:
        print(f"Installing {package}...")
        if not install_package(package):
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\\n❌ Failed to install: {failed_packages}")
        return False
    else:
        print("\\n✅ All packages installed successfully!")
        return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "adapters",
        "notebooks", 
        "data",
        "results"
    ]
    
    print("\\n📁 Creating directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}/")

def verify_huggingface():
    """Verify HuggingFace access"""
    try:
        from huggingface_hub import login
        print("\\n🤗 HuggingFace Hub access:")
        print("You may need to login to access Llama-2 models")
        print("Run: huggingface-cli login")
        print("Get your token from: https://huggingface.co/settings/tokens")
        return True
    except ImportError:
        print("❌ HuggingFace Hub not available")
        return False

def main():
    """Main setup function"""
    print("🏥 AlpaCare Medical Assistant - Setup")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        return
    
    # Check CUDA
    check_cuda()
    
    # Create directories
    create_directories()
    
    # Verify HuggingFace
    verify_huggingface()
    
    print("\\n" + "=" * 50)
    print("🎉 SETUP COMPLETED!")
    print("=" * 50)
    print("\\nNext steps:")
    print("1. Login to HuggingFace: huggingface-cli login")
    print("2. Run training: python train_model.py")
    print("3. Run inference: python inference.py")
    print("4. Or use Jupyter notebooks in notebooks/")
    print("=" * 50)

if __name__ == "__main__":
    main()
