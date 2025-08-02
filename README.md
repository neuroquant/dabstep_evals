# SPOT Evaluation Pipeline Using Inspect


## Quick Start
```

### 1. Set Up the Python Environment

#### Using venv (recommended):

```sh
python3 -m venv spot_env
source spot_env/bin/activate
pip install -r requirements.txt
pip install inspect_ai
pip install git+https://github.com/UKGovernmentBEIS/inspect_evals
```

```

### 2. Run the Pipeline

- inspect eval spot.py --model anthropic/claude-3-5-sonnet-20241022

### 3. See the result

- inspect view start --log-dir "/Users/tommyly/inspect_evals/src/inspect_evals/spot/logs"
