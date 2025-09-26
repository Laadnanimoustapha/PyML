#!/usr/bin/env python3
"""
Runner script to execute all ML examples
"""

import subprocess
import sys
import os

def run_script(script_name):
    """
    Run a Python script and handle errors
    """
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING {script_name.upper()}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully!")
            print("Output:")
            print(result.stdout[-500:])  # Show last 500 characters of output
        else:
            print(f"❌ {script_name} failed with error:")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print(f"⏰ {script_name} timed out after 5 minutes")
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")

def main():
    """
    Main function to run all ML examples
    """
    print("🤖 WELCOME TO THE ULTIMATE ML EXAMPLE SUITE")
    print("=" * 60)
    print("This script will run all machine learning examples in sequence.")
    print("Estimated time: 10-15 minutes depending on your system.")
    print("=" * 60)
    
    # List of scripts to run in order
    scripts = [
        "ml_example.py",
        "advanced_ml_example.py", 
        "neural_network_example.py",
        "comprehensive_ml_suite.py",
        "automl_pipeline.py"
    ]
    
    # Check if all scripts exist
    missing_scripts = []
    for script in scripts:
        if not os.path.exists(script):
            missing_scripts.append(script)
    
    if missing_scripts:
        print(f"❌ Missing scripts: {missing_scripts}")
        print("Please make sure all ML example files are in the current directory.")
        return
    
    # Run each script
    for script in scripts:
        run_script(script)
    
    print("\n" + "=" * 60)
    print("🎉 ALL ML EXAMPLES COMPLETED!")
    print("=" * 60)
    print("\n📚 Summary of what you've learned:")
    print("• Basic ML classification with scikit-learn")
    print("• Advanced ML with multiple algorithms and evaluation techniques")
    print("• Neural networks with TensorFlow/Keras")
    print("• Comprehensive ML suite with clustering and anomaly detection")
    print("• Automated ML pipeline with hyperparameter tuning")
    print("\n🔧 Key techniques covered:")
    print("  - Data preprocessing and feature engineering")
    print("  - Model selection and evaluation")
    print("  - Cross-validation and hyperparameter tuning")
    print("  - Visualization and model interpretability")
    print("  - Classification, regression, clustering, and anomaly detection")
    print("  - Deep learning and automated machine learning")
    print("\n🎓 You now have a comprehensive understanding of ML workflows!")

if __name__ == "__main__":
    main()