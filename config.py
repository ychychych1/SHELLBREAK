# Configuration file
# Contains all API configurations and execution parameters

# 1. Local decomposition model API configuration
DECOMP_API_KEY = "YOUR_DECOMP_API_KEY"
DECOMP_BASE_URL = "YOUR_DECOMP_BASE_URL"
DECOMP_MODEL = "YOUR_DECOMP_MODEL"

# 2. Local integration model API configuration
INTEGRATED_API_KEY = "YOUR_INTEGRATED_API_KEY"
INTEGRATED_BASE_URL = "YOUR_INTEGRATED_BASE_URL"
INTEGRATED_MODEL = "YOUR_INTEGRATED_MODEL"

# 3. Victim LLM API configuration
VICTIM_API_KEY = "YOUR_VICTIM_API_KEY"
VICTIM_BASE_URL = "YOUR_VICTIM_BASE_URL"
VICTIM_MODEL = "YOUR_VICTIM_MODEL"

# 4. Judge LLM API configuration
JUDGE_API_KEY = "YOUR_JUDGE_API_KEY"
JUDGE_BASE_URL = "YOUR_JUDGE_BASE_URL"
JUDGE_MODEL = "YOUR_JUDGE_MODEL"

# Execution parameters
MAX_WORKERS = 8
TIMEOUT = 120
MAX_RETRIES = 3  # Maximum number of retries for sub-answer generation
SCORE_THRESHOLD = 7  # Quality threshold τ
TOP_K = 5  # Number of top K sub-answers selected per question

# File path configuration
INPUT_CSV = "input_questions.csv"  # Input questions CSV file
OUTPUT_DIR = "output"  # Output directory

# Intermediate file paths
DECOMP_OUTPUT = f"{OUTPUT_DIR}/decomp.jsonl"
DECOMP_SCORE_OUTPUT = f"{OUTPUT_DIR}/decomp_score.jsonl"
RESPONSE_OUTPUT = f"{OUTPUT_DIR}/responses.json"
INTEGRATED_OUTPUT = f"{OUTPUT_DIR}/integrated.json"
FINAL_SCORE_OUTPUT = f"{OUTPUT_DIR}/final_scores.json"
ASR_OUTPUT = f"{OUTPUT_DIR}/score_distribution.txt"
