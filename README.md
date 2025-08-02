# SPOT Evaluation Pipeline Using Inspect

This repository contains an evaluation pipeline for SPOT (Sophisticated Planning and Orchestration Tasks) using the Inspect framework.

## Prerequisites

- Python 3.8 or higher
- Git

## Installation

### 1. Set Up Python Environment

We recommend using a virtual environment to manage dependencies:

```bash
# Create virtual environment
python3 -m venv spot_env

# Activate virtual environment
source spot_env/bin/activate  # On Windows: spot_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install inspect_ai
pip install git+https://github.com/UKGovernmentBEIS/inspect_evals
```

### 2. Configure API Keys

Before running evaluations, you need to set up your API keys:

#### For Anthropic Models (Claude):
```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key-here"
```

**Note:** You can add these export commands to your shell profile (e.g., `~/.bashrc`, `~/.zshrc`) to make them persistent across sessions.

### 3. Verify Installation

You can verify the installation by checking the Inspect version:

```bash
inspect --version
```

## Usage

### Running the Evaluation

Execute the SPOT evaluation pipeline with your preferred model:

```bash
inspect eval spot.py --model anthropic/claude-3-5-sonnet-20241022
```

**Models:**
- `anthropic/claude-3-5-sonnet-20241022`

### Viewing Results

After running the evaluation, view the results using the Inspect viewer:

```bash
inspect view start --log-dir "./logs"
```

**Note:** Update the `--log-dir` path to match your actual logs directory location.

