import os
import subprocess
import sys
import argparse
from pathlib import Path

# Import configuration
import config

# Step 1: Question decomposition
def run_decomp():
    print("\n=== Step 1: Decomposing questions ===")
    subprocess.run([sys.executable, 'get_decomp.py', '--input', config.INPUT_CSV, '--output', config.DECOMP_OUTPUT], check=True)

# Step 2: Decomposition scoring
def run_decomp_score():
    print("\n=== Step 2: Scoring decompositions ===")
    subprocess.run([sys.executable, 'get_decomp_score.py', '--input', config.DECOMP_OUTPUT, '--output', config.DECOMP_SCORE_OUTPUT], check=True)

# Step 3: Sub-question responses
def run_response():
    print("\n=== Step 3: Getting sub-question responses ===")
    subprocess.run([sys.executable, 'get_response.py', '--input', config.DECOMP_SCORE_OUTPUT, '--output', config.RESPONSE_OUTPUT], check=True)

# Step 4: Response integration
def run_integrated():
    print("\n=== Step 4: Integrating responses ===")
    subprocess.run([sys.executable, 'get_integrated.py', '--input', config.RESPONSE_OUTPUT, '--output', config.INTEGRATED_OUTPUT], check=True)

# Step 5: Final scoring
def run_final_score():
    print("\n=== Step 5: Scoring integrated responses ===")
    subprocess.run([sys.executable, 'final_integrated_score.py', '--input', config.INTEGRATED_OUTPUT, '--output', config.FINAL_SCORE_OUTPUT], check=True)

# Step 6: Score distribution analysis
def run_asr():
    print("\n=== Step 6: Analyzing score distribution ===")
    subprocess.run([sys.executable, 'ASR.py', '--input', config.FINAL_SCORE_OUTPUT], check=True)

def main():
    parser = argparse.ArgumentParser(description='Run the complete SHELLBREAK pipeline')
    parser.add_argument('--input', help='Input CSV file with questions')
    parser.add_argument('--output_dir', help='Output directory')
    
    args = parser.parse_args()
    
    if args.input:
        config.INPUT_CSV = args.input
    
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
        # Update all output file paths
        config.DECOMP_OUTPUT = f"{config.OUTPUT_DIR}/decomp.jsonl"
        config.DECOMP_SCORE_OUTPUT = f"{config.OUTPUT_DIR}/decomp_score.jsonl"
        config.RESPONSE_OUTPUT = f"{config.OUTPUT_DIR}/responses.json"
        config.INTEGRATED_OUTPUT = f"{config.OUTPUT_DIR}/integrated.json"
        config.FINAL_SCORE_OUTPUT = f"{config.OUTPUT_DIR}/final_scores.json"
        config.ASR_OUTPUT = f"{config.OUTPUT_DIR}/score_distribution.txt"
    
    # Create output directory
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # Check if input file exists
    if not os.path.exists(config.INPUT_CSV):
        print(f"Error: Input file {config.INPUT_CSV} not found.")
        sys.exit(1)
    
    print("Starting SHELLBREAK pipeline...")
    print(f"Input file: {config.INPUT_CSV}")
    print(f"Output directory: {config.OUTPUT_DIR}")
    
    # Run all steps
    run_decomp()
    run_decomp_score()
    run_response()
    run_integrated()
    run_final_score()
    run_asr()
    
    print("\n=== Pipeline completed successfully! ===")
    print(f"Results saved to {config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()
