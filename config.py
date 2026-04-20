# Configuration file
# Contains all API configurations and execution parameters

# 1. Local decomposition model API configuration
DECOMP_API_KEY = "sk-7SifNYgCINhaWSLykqqgUBdmQsF7aY5FpVcwrMmLWa7jP5Xw"
DECOMP_BASE_URL = "https://www.DMXapi.com/v1/chat/completions"
DECOMP_MODEL = "qwen-plus"

# 2. Local integration model API configuration
INTEGRATED_API_KEY = "sk-7SifNYgCINhaWSLykqqgUBdmQsF7aY5FpVcwrMmLWa7jP5Xw"
INTEGRATED_BASE_URL = "https://www.DMXapi.com/v1/chat/completions"
INTEGRATED_MODEL = "qwen-plus"

# 3. Victim LLM API configuration
VICTIM_API_KEY = "sk-7SifNYgCINhaWSLykqqgUBdmQsF7aY5FpVcwrMmLWa7jP5Xw"
VICTIM_BASE_URL = "https://www.DMXapi.com/v1/chat/completions"
VICTIM_MODEL = "qwen-plus"

# 4. Judge LLM API configuration
JUDGE_API_KEY = "sk-7SifNYgCINhaWSLykqqgUBdmQsF7aY5FpVcwrMmLWa7jP5Xw"
JUDGE_BASE_URL = "https://www.DMXapi.com/v1/chat/completions"
JUDGE_MODEL = "qwen-plus"

# Execution parameters
MAX_WORKERS = 8
TIMEOUT = 120
MAX_RETRIES = 3  # Maximum number of retries for sub-answer generation
SCORE_THRESHOLD = 7.5  # Quality threshold τ
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
