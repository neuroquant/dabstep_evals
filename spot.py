import re
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.scorer import choice, model_graded_fact, includes, match, model_graded_qa
from inspect_ai.solver import generate, multiple_choice


ERROR_DETECTION_TEMPLATE = """
You are a scientific peer reviewer. Read this paper and determine if it contains scientific errors.

{question}

Does this paper contain scientific errors that would warrant retraction or correction?

{choices}

Respond with ONLY the letter (A or B). Do not provide any explanation, reasoning, or additional text.
""".strip()


ERROR_CATEGORY_TEMPLATE = """
You are a scientific reviewer conducting a peer review. This paper contains scientific errors. Your job is to identify the specific error category.

{question}

What type of scientific error does this paper contain?

{choices}

Respond with ONLY the error category name. Do not provide any explanation, reasoning, or additional text.

Error Category: """


ERROR_ANNOTATION_TEMPLATE = """
You are a scientific expert conducting peer review. This paper contains a specific scientific error that you must identify.

{question}

Identify the SPECIFIC scientific error in this paper. Focus on:
- Incorrect data, calculations, or values
- Logical gaps or flawed reasoning 
- Methodological problems
- Inconsistent information

Provide a concise, precise explanation of what is wrong (1-2 sentences maximum).

Error: """


def record_to_detection_sample(record: dict[str, Any]) -> Sample:
    """Convert a SPOT record to a detection sample."""
    # Extract text content
    content_parts = []
    for section in record.get("paper_content", []):
        if section.get("type") == "text" and section.get("text"):
            content_parts.append(section["text"])
    
    # Use full content without truncation
    full_content = "\n\n".join(content_parts)
    
    title = record["title"]
    
    return Sample(
        input=f"Title: {title}\n\nContent: {full_content}",
        target="A",  # All papers in SPOT have errors
        choices=["Yes - Contains scientific errors", "No - No scientific errors detected"],
        id=record.get("doi/arxiv_id", "unknown"),
        metadata={
            "true_error_type": record["error_category"],
            "paper_category": record["paper_category"],
            "gt_error_location": record["error_location"],
            "gt_error_annotation": record["error_annotation"]
        }
    )


def record_to_error_category_sample(record: dict[str, Any]) -> Sample:
    """Convert a SPOT record to an error category sample."""
    
    # Extract text content
    content_parts = []
    for section in record.get("paper_content", []):
        if section.get("type") == "text" and section.get("text"):
            content_parts.append(section["text"])
    
    # Use full content without truncation
    full_content = "\n\n".join(content_parts)
    title = record["title"]
    
    # Define standard error category choices
    error_categories = [
        "Equation / proof",
        "Reagent identity", 
        "Figure duplication",
        "Data inconsistency",
        "Experiment setup",
        "Data Inconsistency (figure - text)",
        "Statistical reporting",
        "Data Inconsistency (text-text)"
    ]
    
    return Sample(
        input=f"Title: {title}\n\nContent: {full_content}",
        target=record["error_category"],
        choices=error_categories,
        id=record.get("doi/arxiv_id", "unknown"),
        metadata={
            "paper_category": record["paper_category"],
            "gt_error_category": record["error_category"]
        }
    )


def filter_text_only_records(records: list[dict]) -> list[dict]:
    """Filter records to only include papers with no images (image_url = null)."""
    filtered = []
    for record in records:
        has_images = False
        for section in record.get("paper_content", []):
            if section.get("image_url") is not None:
                has_images = True
                break
        
        if not has_images:
            filtered.append(record)
    
    return filtered


def record_to_annotation_sample(record: dict[str, Any]) -> Sample:
    """Convert a SPOT record to an annotation sample."""
    # Extract text content
    content_parts = []
    for section in record.get("paper_content", []):
        if section.get("type") == "text" and section.get("text"):
            content_parts.append(section["text"])
    
    # Use full content without truncation
    full_content = "\n\n".join(content_parts)
    title = record["title"]
    
    return Sample(
        input=f"Title: {title}\n\nContent: {full_content}",
        target=record["error_annotation"],
        id=record.get("doi/arxiv_id", "unknown"),
        metadata={
            "paper_category": record["paper_category"],
            "gt_error_annotation": record["error_annotation"]
        }
    )



# @task
# def spot() -> Task:
#     """
#     Error detection task - binary classification to identify if papers contain scientific errors
#     """
#     dataset_path = Path(__file__).parent / "spot_dataset.json"
    
#     # Use json_dataset with custom sample conversion
#     dataset = json_dataset(
#         json_file=str(dataset_path),
#         sample_fields=record_to_detection_sample
#     )
    
#     # Limit to first 2 papers for testing (remove limit for full evaluation)
#     limited_dataset = dataset[:2]
    
#     return Task(
#         dataset=limited_dataset,
#         solver=multiple_choice(template=ERROR_DETECTION_TEMPLATE),
#         scorer=includes()
#     )


@task 
def spot_error_category() -> Task:
    """
    Error category task - identify what type of error occurs in papers (text-only, no images)
    """
    import json
    
    dataset_path = Path(__file__).parent / "spot_dataset.json"
    
    # Load and filter JSON data first
    with open(dataset_path, 'r') as f:
        all_records = json.load(f)
    
    # Filter to text-only papers and limit for testing
    text_only_records = filter_text_only_records(all_records)[:5]
    
    # Convert to samples
    samples = [record_to_error_category_sample(record) for record in text_only_records]
    
    return Task(
        dataset=samples,
        solver=multiple_choice(template=ERROR_CATEGORY_TEMPLATE),
        scorer=includes()
    )


@task
def spot_error_annotation() -> Task:
    """
    Error annotation task - explain what the scientific errors are (text-only, no images)
    """
    import json
    
    dataset_path = Path(__file__).parent / "spot_dataset.json"
    
    # Load and filter JSON data first
    with open(dataset_path, 'r') as f:
        all_records = json.load(f)
    
    # Filter to text-only papers and limit for testing
    text_only_records = filter_text_only_records(all_records)[:5]
    
    # Convert to samples
    samples = [record_to_annotation_sample(record) for record in text_only_records]
    
    return Task(
        dataset=samples,
        solver=generate(template=ERROR_ANNOTATION_TEMPLATE),
        scorer=model_graded_qa(
            instructions="""
            You are evaluating whether a model correctly identified a scientific error in a paper.

            Compare the model's error annotation with the expected correct error annotation.

            Score as:
            - GRADE: C (Correct) if the model identifies the same error or a very similar error concept
            - GRADE: P (Partial) if the model identifies a related error but misses key details  
            - GRADE: I (Incorrect) if the model identifies a completely different error or provides irrelevant analysis

            Focus on whether the core error concept matches, not exact wording.
            """,
            model="anthropic/claude-3-5-sonnet-20241022"
        )
    )


