import re
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.scorer import choice, model_graded_fact
from inspect_ai.solver import generate, multiple_choice


ERROR_DETECTION_TEMPLATE = """
You are a scientific peer reviewer. Read this paper and determine if it contains scientific errors.

{question}

Does this paper contain scientific errors that would warrant retraction or correction?

{choices}

Answer with the letter only.
""".strip()


ERROR_LOCATION_TEMPLATE = """
You are a scientific reviewer. This paper contains scientific errors. Identify WHERE the primary error occurs.

{question}

Where in the paper is the main scientific error located? Be specific (e.g., "Theorem 2.1", "Section 3", "Lemma 4", "Equation 5").

Error Location: """


ERROR_ANNOTATION_TEMPLATE = """
You are a scientific expert. This paper contains scientific errors. Provide an annotation explaining what the error is.

{question}

What is the scientific error in this paper? Provide a clear annotation explaining what went wrong.

Error Annotation: """


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


def record_to_location_sample(record: dict[str, Any]) -> Sample:
    """Convert a SPOT record to a location sample."""
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
        target=record["error_location"],
        id=record.get("doi/arxiv_id", "unknown"),
        metadata={
            "paper_category": record["paper_category"],
            "gt_error_location": record["error_location"]
        }
    )


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


def create_negative_control_sample() -> Sample:
    """Create a negative control sample with an obviously wrong statement."""
    return Sample(
        input="Title: A Simple Mathematical Truth\n\nContent: In this paper, we prove that 2 + 2 = 5. This is a fundamental result in arithmetic that has been overlooked by mathematicians for centuries. Our proof is as follows: Let a = 2 and b = 2. Then a + b = 2 + 2. Through careful analysis, we conclude that 2 + 2 = 5. This completes our proof.",
        target="A",  # Should contain scientific errors
        choices=["Yes - Contains scientific errors", "No - No scientific errors detected"],
        id="negative_control",
        metadata={
            "true_error_type": "Mathematical error",
            "paper_category": "Mathematics",
            "gt_error_location": "Proof section",
            "gt_error_annotation": "The claim that 2 + 2 = 5 is mathematically incorrect. Basic arithmetic shows that 2 + 2 = 4."
        }
    )


def create_negative_control_location_sample() -> Sample:
    """Create a negative control sample for location task."""
    return Sample(
        input="Title: A Simple Mathematical Truth\n\nContent: In this paper, we prove that 2 + 2 = 5. This is a fundamental result in arithmetic that has been overlooked by mathematicians for centuries. Our proof is as follows: Let a = 2 and b = 2. Then a + b = 2 + 2. Through careful analysis, we conclude that 2 + 2 = 5. This completes our proof.",
        target="Proof section",
        id="negative_control_location",
        metadata={
            "paper_category": "Mathematics",
            "gt_error_location": "Proof section"
        }
    )


def create_negative_control_annotation_sample() -> Sample:
    """Create a negative control sample for annotation task."""
    return Sample(
        input="Title: A Simple Mathematical Truth\n\nContent: In this paper, we prove that 2 + 2 = 5. This is a fundamental result in arithmetic that has been overlooked by mathematicians for centuries. Our proof is as follows: Let a = 2 and b = 2. Then a + b = 2 + 2. Through careful analysis, we conclude that 2 + 2 = 5. This completes our proof.",
        target="The claim that 2 + 2 = 5 is mathematically incorrect. Basic arithmetic shows that 2 + 2 = 4.",
        id="negative_control_annotation",
        metadata={
            "paper_category": "Mathematics",
            "gt_error_annotation": "The claim that 2 + 2 = 5 is mathematically incorrect. Basic arithmetic shows that 2 + 2 = 4."
        }
    )


@task
def simple_spot() -> Task:
    """
    Error detection task - binary classification to identify if papers contain scientific errors
    """
    dataset_path = Path(__file__).parent / "spot_dataset.json"
    
    # Use json_dataset with custom sample conversion
    dataset = json_dataset(
        json_file=str(dataset_path),
        sample_fields=record_to_detection_sample
    )
    
    # Limit to first 2 papers for testing (remove limit for full evaluation)
    limited_dataset = dataset[:2]
    
    # Add negative control sample
    control_sample = create_negative_control_sample()
    combined_samples = list(limited_dataset) + [control_sample]
    
    return Task(
        dataset=combined_samples,
        solver=multiple_choice(template=ERROR_DETECTION_TEMPLATE),
        scorer=choice()
    )


@task 
def simple_spot_location() -> Task:
    """
    Error location task - identify where errors occur in papers
    """
    dataset_path = Path(__file__).parent / "spot_dataset.json"
    
    # Use json_dataset with custom sample conversion
    dataset = json_dataset(
        json_file=str(dataset_path),
        sample_fields=record_to_location_sample
    )
    
    # Limit to first 2 papers for testing
    limited_dataset = dataset[:2]
    
    # Add negative control sample
    control_sample = create_negative_control_location_sample()
    combined_samples = list(limited_dataset) + [control_sample]
    
    return Task(
        dataset=combined_samples,
        solver=generate(template=ERROR_LOCATION_TEMPLATE),
        scorer=choice()
    )


@task
def simple_spot_annotation() -> Task:
    """
    Error annotation task - explain what the scientific errors are
    """
    dataset_path = Path(__file__).parent / "spot_dataset.json"
    
    # Use json_dataset with custom sample conversion
    dataset = json_dataset(
        json_file=str(dataset_path),
        sample_fields=record_to_annotation_sample
    )
    
    # Limit to first 2 papers for testing
    limited_dataset = dataset[:2]
    
    # Add negative control sample
    control_sample = create_negative_control_annotation_sample()
    combined_samples = list(limited_dataset) + [control_sample]
    
    return Task(
        dataset=combined_samples,
        solver=generate(template=ERROR_ANNOTATION_TEMPLATE),
        scorer=model_graded_fact(
            model="anthropic/claude-3-5-sonnet-20241022"
        )
    )


