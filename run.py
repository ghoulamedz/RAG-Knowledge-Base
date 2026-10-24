#!/usr/bin/env python3
"""
Quick start script for the RAG Knowledge Base.
Starts the FastAPI server on port 8000.
"""

import subprocess
import sys
import time
from pathlib import Path

API_DIR = Path(__file__).parent / "api"
API_SCRIPT = API_DIR / "server.py"
DATA_DIR = Path(__file__).parent / "data"

def main():
    print("=" * 60)
    print("  RAG Knowledge Base — Starting...")
    print("=" * 60)
    print()
    
    # Check Ollama
    print("📦 Checking Ollama...")
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=False
        )
        models = [line.split()[0] for line in result.stdout.split("\n") if line.strip()]
        if models:
            print(f"   ✓ Ollama available. Models: {', '.join(models)}")
        else:
            print("   ⚠ Ollama not found. Install with: curl -fsSL https://ollama.com/install.sh | sh")
            print("   Then run: ollama pull llama3.2")
    except FileNotFoundError:
        print("   ⚠ Ollama not found. Install with: curl -fsSL https://ollama.com/install.sh | sh")
    
    print()
    
    # Start the API server
    print("🚀 Starting API server on http://localhost:8000")
    print()
    
    try:
        process = subprocess.run(
            [sys.executable, str(API_SCRIPT)],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Server exited with code {e.returncode}")
        print(f"   stderr: {e.stderr.decode() if e.stderr else 'N/A'}")
        sys.exit(1)

if __name__ == "__main__":
    main()
