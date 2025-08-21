import re
import json
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.scorer import choice, model_graded_fact, includes, match, model_graded_qa, f1, exact, scorer
from inspect_ai.solver import generate, multiple_choice
from inspect_ai.model import GenerateConfig


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
Find the specific scientific error in this paper. Do not summarize.

Example (format only):
Error: Misapplied lemma: Lemma 2.1 requires X > 0, but the proof uses X < 0.

{question}

Respond with exactly one line starting with "Error: " followed by the specific error (less than 50 words). Do not include any other text.
""".strip()


# Original SPOT paper prompts (exact copy from their prompts.py)
ORIGINAL_SPOT_REVIEWER_TEMPLATE = """You are a **scientific‐rigor auditor**. You will receive the parsed contents of a research paper. Your job is to identify **only** those errors or flaws that directly undermine the **scientific validity** of the paper's methods, analyses, or conclusions.

Your sole focus is to identify flaws—such as errors in experimental design, data integrity, calculations, statistical inference, or reproducibility—that directly call into question the validity of a specific claim, paragraph, or the paper as a whole.

**Do not** report issues purely presentational, rhetorical, stylistic, or related to citation practices.

---

{question}

After you've done a **detailed walkthrough** of the paper, identify any scientific errors you find. For each error, specify:

1. **Location**: Where in the paper the error occurs (e.g., "Section 2.1", "Figure 3", "Equation 5")
2. **Description**: A clear explanation of what the scientific error is and why it undermines the paper's validity

Examples of scientific errors you might identify:

- **Section 2.1**: Claim that 'all X are Y' is contradicted by citation [5], which shows examples of non‑Y X.
- **Figure 3**: XAxis labeled 'Time (s)' but units appear to be milliseconds; scale bar mismatches caption.
- **Equation 5**: Misapplied lemma: Lemma 2.1 requires X > 0, but the proof uses X < 0.
- **Statistical Analysis**: P-value calculation incorrect due to wrong degrees of freedom in t-test.

Be as precise as possible about **where** (section ID or figure/table) and **what** the scientific flaw is. Each description must be rooted in a scientific rationale explaining why it is 'wrong' (not how it could be improved).

Begin your analysis now.""".strip()




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
    # Use full content without character limit
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


# @task 
# def spot_error_category() -> Task:
#     """
#     Error category task - identify what type of error occurs in papers (text-only, no images)
#     """
#     import json
    
#     dataset_path = Path(__file__).parent / "spot_dataset.json"
    
#     # Load and filter JSON data first
#     with open(dataset_path, 'r') as f:
#         all_records = json.load(f)
    
#     # Filter to text-only papers and limit for testing
#     text_only_records = filter_text_only_records(all_records)[:5]
    
#     # Convert to samples
#     samples = [record_to_error_category_sample(record) for record in text_only_records]
    
#     return Task(
#         dataset=samples,
#         solver=multiple_choice(template=ERROR_CATEGORY_TEMPLATE),
#         scorer=includes()
#     )


# @task
# def spot_error_annotation() -> Task:
#     """
#     Error annotation task - explain what the scientific errors are (text-only, no images)
#     """
#     import json
    
#     dataset_path = Path(__file__).parent / "spot_dataset.json"
    
#     # Load and filter JSON data first
#     with open(dataset_path, 'r') as f:
#         all_records = json.load(f)
    
#     # Filter to text-only papers and limit for testing
#     text_only_records = filter_text_only_records(all_records)[10:15]
    
#     # Convert to samples
#     samples = [record_to_annotation_sample(record) for record in text_only_records]
    
#     return Task(
#         dataset=samples,
#         solver=generate(template=ERROR_ANNOTATION_TEMPLATE),
#         config=GenerateConfig(
#             temperature=0.0,
#             max_tokens=256,
#             system_message="You are an error detector. Find the specific scientific error in the paper. Do not summarize. The paper contains a scientific error - identify it specifically."
#         ),
#         scorer=model_graded_qa(
#             instructions="""
#             You are evaluating whether a model correctly identified a scientific error in a paper.
            
#             Compare the model's error identification with the expected error. Score based on:
#             - Does the model identify the same core error concept?
#             - Is the model's explanation scientifically accurate?
#             - Does it match the key points in the expected answer?
            
#             Use this rubric:
#             GRADE: C (if the model identifies the same error or very similar concept)
#             GRADE: P (if the model identifies a related error but misses key details)
#             GRADE: I (if the model identifies the wrong error or provides a general summary instead)
            
#             Only reply with a single letter: C, P, or I.
#             """,
#             grade_pattern=r"\b([CPI])\b",
#             partial_credit=True,
#             model="openai/gpt-4o-mini"
#         )
#     )


@task
def spot_original_methodology() -> Task:
    """
    SPOT evaluation using the original paper's methodology with structured JSON output.
    This task uses the exact prompts and evaluation approach from the original SPOT paper.
    """
    import json
    
    dataset_path = Path(__file__).parent / "spot_dataset.json"
    
    # Load and filter JSON data first
    with open(dataset_path, 'r') as f:
        all_records = json.load(f)
    
    # Filter to text-only papers and limit for testing
    text_only_records = filter_text_only_records(all_records)[10:15]
    
    # Convert to samples using the original SPOT format
    samples = [record_to_annotation_sample(record) for record in text_only_records]
    
    return Task(
        dataset=samples,
        solver=generate(),
        config=GenerateConfig(
            temperature=0.0,
            max_tokens=1024,
            system_message="""You are a **scientific‐rigor auditor**. Your job is to identify **only** those errors or flaws that directly undermine the **scientific validity** of the paper's methods, analyses, or conclusions.

                        Your sole focus is to identify flaws—such as errors in experimental design, data integrity, calculations, statistical inference, or reproducibility—that directly call into question the validity of a specific claim, paragraph, or the paper as a whole.

                        **Do not** report issues purely presentational, rhetorical, stylistic, or related to citation practices.

                        After you've done a **detailed walkthrough** of the paper, identify any scientific errors you find. For each error, specify:

                        1. **Location**: Where in the paper the error occurs (e.g., "Section 2.1", "Figure 3", "Equation 5")
                        2. **Description**: A clear explanation of what the scientific error is and why it undermines the paper's validity

                        Examples of scientific errors you might identify:

                        - **Section 2.1**: Claim that 'all X are Y' is contradicted by citation [5], which shows examples of non‑Y X.
                        - **Figure 3**: XAxis labeled 'Time (s)' but units appear to be milliseconds; scale bar mismatches caption.
                        - **Equation 5**: Misapplied lemma: Lemma 2.1 requires X > 0, but the proof uses X < 0.
                        - **Statistical Analysis**: P-value calculation incorrect due to wrong degrees of freedom in t-test.

                        Be as precise as possible about **where** (section ID or figure/table) and **what** the scientific flaw is. Each description must be rooted in a scientific rationale explaining why it is 'wrong' (not how it could be improved).

                        Focus only on scientific validity issues, not presentational, rhetorical, or citation problems."""
        ),
        scorer=model_graded_qa(
            instructions="""
            You are evaluating whether a model correctly identified scientific errors using the original SPOT methodology.
            
            The model should identify scientific errors with:
            1. **Location**: Where in the paper the error occurs (e.g., "Section 2.1", "Figure 3", "Equation 5")
            2. **Description**: A clear explanation of what the scientific error is and why it undermines the paper's validity
            
            Score based on:
            - Does the model identify scientific errors that directly undermine the paper's scientific validity?
            - Does the model provide specific locations and clear descriptions of the errors?
            - Does the model focus only on scientific validity issues (not presentational, rhetorical, or citation issues)?
            - Does the model provide scientific rationale explaining why the errors are 'wrong'?
            
            Use this rubric:
            GRADE: C (if the model correctly identifies scientific errors with specific locations and descriptions)
            GRADE: P (if the model identifies errors but locations or descriptions are vague or incomplete)
            GRADE: I (if the model doesn't identify errors or focuses on non-scientific issues)
            
            Only reply with a single letter: C, P, or I.
            """,
            grade_pattern=r"\b([CPI])\b",
            partial_credit=True,
            model="openai/gpt-4o-mini"
        )
    )


@task
def spot_original_methodology_f1() -> Task:
    """
    SPOT evaluation using F1 scoring for text similarity.
    This task uses the original SPOT prompts but with F1 scoring instead of model-graded evaluation.
    """
    import json
    
    dataset_path = Path(__file__).parent / "spot_dataset.json"
    
    # Load and filter JSON data first
    with open(dataset_path, 'r') as f:
        all_records = json.load(f)
    
    # Filter to text-only papers and limit for testing
    text_only_records = filter_text_only_records(all_records)[10:15]
    
    # Convert to samples using the original SPOT format
    samples = [record_to_annotation_sample(record) for record in text_only_records]
    
    return Task(
        dataset=samples,
        solver=generate(),
        config=GenerateConfig(
            temperature=0.0,
            max_tokens=1024,
            system_message="""You are a **scientific‐rigor auditor**. Your job is to identify **only** those errors or flaws that directly undermine the **scientific validity** of the paper's methods, analyses, or conclusions.

                        Your sole focus is to identify flaws—such as errors in experimental design, data integrity, calculations, statistical inference, or reproducibility—that directly call into question the validity of a specific claim, paragraph, or the paper as a whole.

                        **Do not** report issues purely presentational, rhetorical, stylistic, or related to citation practices.

                        After you've done a **detailed walkthrough** of the paper, identify any scientific errors you find. For each error, specify:

                        1. **Location**: Where in the paper the error occurs (e.g., "Section 2.1", "Figure 3", "Equation 5")
                        2. **Description**: A clear explanation of what the scientific error is and why it undermines the paper's validity

                        Examples of scientific errors you might identify:
                        - **Section 2.1**: Claim that 'all X are Y' is contradicted by citation [5], which shows examples of non‑Y X.
                        - **Figure 3**: XAxis labeled 'Time (s)' but units appear to be milliseconds; scale bar mismatches caption.
                        - **Equation 5**: Misapplied lemma: Lemma 2.1 requires X > 0, but the proof uses X < 0.
                        - **Statistical Analysis**: P-value calculation incorrect due to wrong degrees of freedom in t-test.

                        Be as precise as possible about **where** (section ID or figure/table) and **what** the scientific flaw is. Each description must be rooted in a scientific rationale explaining why it is 'wrong' (not how it could be improved).

                        Focus only on scientific validity issues, not presentational, rhetorical, or citation problems."""
        ),
        scorer=f1()
    )


