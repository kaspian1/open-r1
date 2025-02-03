from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def test_gpu():
    # Check CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

if __name__ == "__main__":
    test_gpu()