#!/usr/bin/env python3
"""
Demo runner script for Financial Document Analysis.

This script provides an easy way to run the Streamlit demo application
and other demonstration features.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_streamlit_demo():
    """Run the Streamlit demo application."""
    demo_path = Path(__file__).parent.parent / "demo" / "app.py"
    
    if not demo_path.exists():
        print(f"Error: Demo file not found at {demo_path}")
        return False
    
    print("Starting Financial Document Analysis Demo...")
    print("The demo will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the demo")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(demo_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Error running demo: {e}")
        return False
    
    return True


def run_training_demo():
    """Run a quick training demonstration."""
    script_path = Path(__file__).parent.parent / "scripts" / "train_models.py"
    
    if not script_path.exists():
        print(f"Error: Training script not found at {script_path}")
        return False
    
    print("Running training demonstration...")
    
    try:
        subprocess.run([
            sys.executable, str(script_path),
            "--output-dir", "assets",
            "--verbose"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running training: {e}")
        return False
    
    return True


def run_tests():
    """Run the test suite."""
    test_path = Path(__file__).parent.parent / "tests"
    
    if not test_path.exists():
        print(f"Error: Test directory not found at {test_path}")
        return False
    
    print("Running test suite...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pytest", str(test_path),
            "-v", "--tb=short"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running tests: {e}")
        return False
    
    return True


def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        "streamlit", "pandas", "numpy", "spacy", "transformers",
        "scikit-learn", "plotly", "beautifulsoup4", "PyPDF2",
        "pdfplumber", "python-docx"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print("pip install -r requirements.txt")
        return False
    
    print("All required dependencies are installed!")
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Financial Document Analysis Demo Runner"
    )
    parser.add_argument(
        "command",
        choices=["demo", "train", "test", "check"],
        help="Command to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    print("Financial Document Analysis Demo Runner")
    print("=" * 40)
    
    success = False
    
    if args.command == "demo":
        success = run_streamlit_demo()
    elif args.command == "train":
        success = run_training_demo()
    elif args.command == "test":
        success = run_tests()
    elif args.command == "check":
        success = check_dependencies()
    
    if success:
        print("\nCommand completed successfully!")
        sys.exit(0)
    else:
        print("\nCommand failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
