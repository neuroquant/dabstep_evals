#!/usr/bin/env python3
"""
SPOT Evaluation Pipeline
========================
This script evaluates LLM performance on the SPOT (Scientific Paper Error Detection) dataset
using Inspect AI framework.
"""

#%%
# IMPORTS AND SETUP
import os
import json
import re
import glob
import tempfile
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Dataset, Sample, MemoryDataset, json_dataset, FieldSpec
from inspect_ai.model import get_model
from inspect_ai.solver import generate
from inspect_ai.scorer import scorer, Score, accuracy, stderr
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.scorer import model_graded_fact
from datasets import load_dataset
import pandas as pd
from inspect_ai.scorer import score as score_log
from inspect_ai.scorer import Scorer, Target
from inspect_ai._eval.task.run import TaskState


#%% Load dataset and convert to JSON
dataset = load_dataset("amphora/SPOT", split="train")

data = [dict(row) for row in dataset]
with open("spot_dataset.json", "w") as f:
    json.dump(data, f, indent=2)

print(f" Downloaded {len(data)} samples from SPOT dataset")

#% Define scorer
@scorer(metrics=[accuracy(), stderr()])
def spot_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        # Extract ground truth info from target
        target_text = target.text
        gt_error_location = ""
        gt_error_annotation = ""
        errors_detected = False
        
        # Parse ground truth from target text
        if "Error Location:" in target_text:
            lines = target_text.split('\n')
            for line in lines:
                if line.startswith("- Error Location:"):
                    gt_error_location = line.replace("- Error Location:", "").strip()
                elif line.startswith("- Error Annotation:"):
                    gt_error_annotation = line.replace("- Error Annotation:", "").strip()
                elif line.startswith("- Errors Detected:"):
                    errors_detected = line.replace("- Errors Detected:", "").strip().lower() == "true"
        
        # Get model output
        model_output = str(state.output)
        
        # Create grading prompt
        grading_prompt = f"""
You are evaluating whether a model correctly identified scientific errors in a paper.

The model was asked to analyze a paper and identify any scientific errors, issues, or problems. 
The model provided:
{model_output}

GROUND TRUTH:
- error_location: {gt_error_location}
- error_annotation: {gt_error_annotation}
- errors_detected: {errors_detected}

Your task is to determine if the model's error detection and description match the ground truth.

Consider:
1. Did the model find an error when ground truth says there should be one?
2. Does the model's error location match or overlap with the ground truth location?
3. Does the model's error description match or align with the ground truth annotation?

Respond with:
GRADE: C (Correct) - if the model's findings reasonably match the ground truth
GRADE: I (Incorrect) - if the model's findings don't match the ground truth

Provide your reasoning for the grade.
"""
        
        # Use the same model for grading
        grading_model = get_model("anthropic/claude-3-5-haiku-20241022")
        response = await grading_model.generate(grading_prompt)
        
        # Extract grade from response
        grade = "I"  # Default to incorrect
        explanation = str(response)
        
        if "GRADE: C" in str(response).upper():
            grade = "C"
        elif "GRADE: I" in str(response).upper():
            grade = "I"
        
        return Score(value=grade, explanation=explanation)
    
    return score


#%% Setup API key
# Get API key from environment variable or use the provided key
api_key = os.getenv('ANTHROPIC_API_KEY', 'INPUT-YOUR-API-KEY')
os.environ['ANTHROPIC_API_KEY'] = api_key
print("✅ API key set")

#%% Process data and create dataset

with open("spot_dataset.json") as f:
    raw_data = json.load(f)

print(f"Loaded {len(raw_data)} samples from JSON file")

# Define field mapping
field_spec = FieldSpec(
    input="prompt",
    target="target", 
    id="paper_id"
)

print("Field spec defined")

#% Process samples (just first 3 for testing)
processed_data = []
for i, row in enumerate(raw_data[:3]):  # Process only 3 samples for testing
    paper_content = row["paper_content"]
    if isinstance(paper_content, list):
        # Extract text from each chunk and join
        paper_text = '\n\n'.join(
            chunk['text'] for chunk in paper_content 
            if isinstance(chunk, dict) and chunk.get('text')
        )
    else:
        paper_text = str(paper_content)
    
    # Use the entire paper text without truncation
    print(f"Sample {i}: Using full paper ({len(paper_text)} chars)")
    
    prompt = f"""
You are a scientific rigor auditor specializing in academic paper verification. 

PAPER CONTENT:
{paper_text}

Your task is to analyze this paper and identify any scientific errors, issues, or problems. Look for:
1. Methodological errors
2. Statistical errors  
3. Logical inconsistencies
4. Data interpretation errors
5. Citation or reference errors
6. Any other scientific issues

Please respond in this exact JSON format:
{{
    "error_location": "specific location where you found the error (e.g., 'Figure 2 caption', 'Methods section paragraph 3', 'Results section line 45')",
    "error_annotation": "detailed description of the error you found"
}}

Respond only with the JSON object, no additional text.
"""
    
    # Create target
    gt_error_location = row.get('error_location', '')
    gt_error_annotation = row.get('error_annotation', '')
    target_text = f"""
GROUND TRUTH ERROR INFORMATION:
- Error Location: {gt_error_location}
- Error Annotation: {gt_error_annotation}
- Errors Detected: {row.get("errors_detected", False)}

The model should have identified this specific error location and provided a description that matches the ground truth annotation.
"""
    
    processed_row = {
        "prompt": prompt,
        "target": target_text,
        "paper_id": f"paper_{i}"
    }
    processed_data.append(processed_row)

print(f"Processed {len(processed_data)} samples")

# Save to temporary JSONL file
temp_file = "temp_processed_data.jsonl"
with open(temp_file, "w") as f:
    for row in processed_data:
        f.write(json.dumps(row) + "\n")

print(f"Saved processed data to {temp_file}")

#%% Use json_dataset from inspect_ai
try:
    dataset = json_dataset(temp_file, field_spec)
    print(f"JSON Dataset created with {len(dataset)} samples")
    
except Exception as e:
    print(f"Error creating dataset: {e}")

#%% Create task and model
try:
    model = get_model("anthropic/claude-3-5-haiku-20241022")
    print("Model loaded successfully")
    
    task = Task(
        dataset=dataset,
        solver=generate(),
        name="spot_inference_task",
        version=1
    )
    print("Task created successfully")
    
except Exception as e:
    print(f"Error creating task/model: {e}")

#%% Run evaluation
log_dir = tempfile.mkdtemp(prefix="inspect_ai_logs_")
results = inspect_eval(
    tasks=task,
    model=model,
    limit=len(processed_data),
    display="log",
    log_dir=log_dir,
    log_format="json",
    log_level="info"
)
print(f"✅ Evaluation complete. Log directory: {log_dir}")

#%% Score the results
log_files = glob.glob(os.path.join(log_dir, "*.json"))
if log_files:
    log_file = log_files[0]    
    # Read the log
    log = read_eval_log(log_file)
    print(f"Read log with {len(log.samples)} samples")
    print("Scoring results with native scorer...")
    
    # Apply the native scorer to each sample
    import asyncio
    
    async def score_samples():
        for i, sample in enumerate(log.samples):
            print(f"🎯 Scoring sample {i+1}...")
            
            try:
                # Create TaskState-like object
                class SimpleTaskState:
                    def __init__(self, sample):
                        self.output = sample.output
                        self.input = sample.input
                        self.id = sample.id
                
                # Create Target from sample target
                target = Target([sample.target])
                
                # Create state
                state = SimpleTaskState(sample)
                
                # Apply the scorer
                score_result = await spot_scorer()(state, target)
                print(f"   ✅ Score result: {score_result}")
                
                # Add the score to the sample
                if not hasattr(sample, 'scores'):
                    sample.scores = []
                elif isinstance(sample.scores, dict):
                    # Convert dict to list if needed
                    sample.scores = [sample.scores]
                sample.scores.append(score_result)
                
            except Exception as e:
                print(f"   ❌ Error scoring sample {i+1}: {e}")
                # Add a default score
                if not hasattr(sample, 'scores'):
                    sample.scores = []
                sample.scores.append({"value": "ERROR", "explanation": str(e)})
    
    # Run the async scoring
    asyncio.run(score_samples())
    scored_log = log
    
    # Save scored results
    base, ext = os.path.splitext(log_file)
    scored_file = base + "_scored" + ext
    write_eval_log(scored_log, scored_file)
    print(f"✅ Scored results saved to: {scored_file}")
    
else:
    print("❌ No log files found for scoring")


