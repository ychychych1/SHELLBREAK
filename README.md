# SSG-GRPO Pipeline

## Project Overview

This project implements the SSG-GRPO (Semantic Safety-Guarded Reward Policy Optimization) pipeline for question decomposition and evaluation. The system decomposes complex questions into sub-questions, evaluates safety and quality, generates answers, and produces final integrated responses.

## Project Structure

| File | Description |
|------|-------------|
| `config.py` | Central configuration file managing API keys, model endpoints, file paths, and execution parameters |
| `main.py` | Pipeline entry point that orchestrates the entire workflow |
| `get_decomp.py` | Decomposes original questions into sub-questions using the local decomposition model API |
| `get_decomp_score.py` | Scores and cleans decomposed sub-questions using Judge LLM API |
| `get_response.py` | Generates answers for sub-questions using Victim LLM API |
| `get_integrated.py` | Integrates sub-answers using local integration model API |
| `final_integrated.py` | Performs final scoring of integrated answers using Judge LLM API |
| `ASR.py` | Calculates Acceptance Success Rate (ASR) statistics |
| `SSG-GRPO_CORE.py` | Core implementation of SSG-GRPO reward function and DFI (Decomposition Faithfulness Index) calculation |
| `question.csv` | Input file containing questions to process |

## Pipeline Flow

```
question.csv → get_decomp.py → get_decomp_score.py → get_response.py
                                                    ↓
                                            get_integrated.py
                                                    ↓
                                          final_integrated.py
                                                    ↓
                                                 ASR.py
```

## Configuration

Before running, edit `config.py` to set:

```python
# API Configuration
JUDGE_API_KEY = "your-judge-api-key"
JUDGE_BASE_URL = "https://your-judge-model-endpoint"

VICTIM_API_KEY = "your-victim-api-key"
VICTIM_BASE_URL = "https://your-victim-model-endpoint"

DECOMP_API_KEY = "your-decomp-api-key"
DECOMP_BASE_URL = "https://your-decomp-model-endpoint"

INTEGRATED_API_KEY = "your-integrated-api-key"
INTEGRATED_BASE_URL = "https://your-integrated-model-endpoint"

# File Paths
INPUT_CSV = "question.csv"
OUTPUT_DIR = "output"
```

## Usage

### Run Full Pipeline

```bash
python main.py
```

### Run Individual Steps

```bash
# Step 1: Question Decomposition
python get_decomp.py

# Step 2: Decomposition Scoring
python get_decomp_score.py

# Step 3: Sub-question Answer Generation
python get_response.py

# Step 4: Answer Integration
python get_integrated.py

# Step 5: Final Integrated Scoring
python final_integrated.py

# Step 6: ASR Statistics
python ASR.py
```

## Output Files

- `output/decomp.jsonl` - Decomposed sub-questions
- `output/decomp_score.jsonl` - Scored and cleaned sub-questions
- `output/sub_answers.jsonl` - Generated sub-answers
- `output/integrated.jsonl` - Integrated answers
- `output/final_score.jsonl` - Final scores
- `output/asr_result.json` - ASR statistics

## SSG-GRPO Reward Function

The `SSG-GRPO_CORE.py` provides the core reward calculation:

- **DFI (Decomposition Faithfulness Index)**: Measures semantic alignment between original and decomposed questions using attention-weighted embedding aggregation
- **Safety Assessment**: Evaluates sub-question safety and applies safety adjustments
- **Auxiliary Evaluation**: Assesses clarity, granularity, and non-redundancy

## Requirements

```
numpy
requests
```

Install dependencies:

```bash
pip install numpy requests
```
