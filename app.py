import hashlib
import html
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import Literal


load_dotenv()

st.set_page_config(
    page_title="AI Resume Screening Assistant",
    page_icon="📄",
    layout="wide",
)

st.markdown(
    """
    <style>
        :root {
            --bottle: #0B4F45;
            --bottle2: #14695B;
            --mint: #EAF4F0;
            --mint2: #D9EBE4;
            --sage: #F5F8F7;
            --border: #D6E2DE;
            --text: #22302C;
            --muted: #687570;
            --white: #FFFFFF;
        }

        .stApp {
            background: linear-gradient(180deg, #F9FBFA 0%, #F4F7F6 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 1450px;
            padding-top: 1rem;
            padding-bottom: 3rem;
        }

        html, body, p, li, label,
        div[data-testid="stMarkdownContainer"],
        div[data-testid="stWidgetLabel"] p,
        textarea, input {
            font-size: 18px !important;
        }

        h1 {font-size: 2.4rem !important;}
        h2 {font-size: 2rem !important;}
        h3 {
            font-size: 1.55rem !important;
            color: var(--bottle) !important;
            font-weight: 800 !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 1.35rem !important;
            border-bottom: 1px solid #DDE6E3 !important;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 0.95rem 0.35rem 0.75rem 0.35rem !important;
            border-bottom: 3px solid transparent !important;
        }

        .stTabs [data-baseweb="tab"] p,
        .stTabs [data-baseweb="tab"] span {
            color: var(--bottle) !important;
            font-size: 1.35rem !important;
            font-weight: 800 !important;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            border-bottom: 3px solid var(--bottle) !important;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] p,
        .stTabs [data-baseweb="tab"][aria-selected="true"] span {
            color: var(--bottle) !important;
        }

        .stTabs [data-baseweb="tab-highlight"] {
            background-color: var(--bottle) !important;
        }

        div.stButton > button,
        button[data-testid="stBaseButton-primary"],
        button[kind="primary"] {
            background: var(--bottle) !important;
            border: 1px solid var(--bottle) !important;
            color: #FFFFFF !important;
            border-radius: 12px !important;
            min-height: 48px !important;
            padding: 0.65rem 1.25rem !important;
            font-size: 1.05rem !important;
            font-weight: 800 !important;
            box-shadow: 0 5px 16px rgba(11,79,69,.12) !important;
        }

        div.stButton > button:hover,
        div.stButton > button:active,
        div.stButton > button:focus,
        button[data-testid="stBaseButton-primary"]:hover,
        button[data-testid="stBaseButton-primary"]:active,
        button[data-testid="stBaseButton-primary"]:focus,
        button[kind="primary"]:hover,
        button[kind="primary"]:active,
        button[kind="primary"]:focus {
            background: var(--bottle2) !important;
            border-color: var(--bottle2) !important;
            color: #FFFFFF !important;
            box-shadow: 0 0 0 3px rgba(11,79,69,.15) !important;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(145deg, #FFFFFF 0%, #F4F8F6 100%);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: 0 7px 22px rgba(11,79,69,.07);
        }

        div[data-testid="stMetricLabel"] p {
            font-size: 1rem !important;
            color: var(--muted) !important;
            font-weight: 700 !important;
        }

        div[data-testid="stMetricValue"] {
            color: var(--bottle) !important;
        }

        [data-testid="stProgressBar"] > div > div {
            background-color: var(--bottle) !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            border-radius: 16px;
            border-color: var(--border);
            background: #F1F5F4;
        }

        .result-card {
            background: linear-gradient(145deg, #FFFFFF 0%, #F4F8F6 100%);
            border: 1px solid #D8E5E0;
            border-top: 5px solid var(--bottle);
            border-radius: 20px;
            padding: 22px 24px;
            margin: 4px 0 20px 0;
            box-shadow: 0 8px 24px rgba(11,79,69,.08);
        }

        .result-card h4 {
            margin: 0 0 14px 0;
            color: var(--bottle);
            font-size: 1.34rem;
            font-weight: 850;
        }

        .result-item {
            display: flex;
            gap: 10px;
            align-items: flex-start;
            padding: 10px 0;
            border-bottom: 1px solid rgba(216,229,224,.75);
            line-height: 1.5;
            font-size: 1rem;
        }

        .result-item:last-child {
            border-bottom: 0;
        }

        .tick {
            color: var(--bottle);
            font-weight: 900;
        }

        .gap {
            color: #6C7773;
            font-weight: 900;
        }

        .recommendation-card {
            background: linear-gradient(135deg, #DCEFE7 0%, #F9FCFB 100%);
            border: 1px solid #BCD8CE;
            border-left: 7px solid var(--bottle);
            border-radius: 20px;
            padding: 24px 26px;
            margin: 10px 0 20px 0;
            box-shadow: 0 9px 26px rgba(11,79,69,.09);
        }

        .recommendation-label {
            color: #66736F;
            font-size: .9rem;
            font-weight: 850;
            text-transform: uppercase;
            letter-spacing: .055em;
        }

        .recommendation-value {
            color: var(--bottle);
            font-size: 1.6rem;
            line-height: 1.2;
            font-weight: 900;
            margin: 5px 0 12px 0;
        }

        .recommendation-text {
            color: #28342F;
            font-size: 1.03rem;
            line-height: 1.65;
        }

        .best-fit-card {
            background: linear-gradient(135deg, #E2F1EB 0%, #FFFFFF 100%);
            border: 1px solid #C4DDD4;
            border-radius: 20px;
            padding: 22px 24px;
            box-shadow: 0 8px 24px rgba(11,79,69,.08);
        }

        .caption-note {
            color: var(--muted);
            font-size: .95rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None


API_KEY = get_api_key()
if API_KEY:
    os.environ["OPENAI_API_KEY"] = API_KEY


class CandidateEvaluation(BaseModel):
    candidate_name: str
    match_score: float = Field(ge=0, le=100)
    matching_skills: list[str]
    missing_skills: list[str]
    candidate_summary: str
    strengths: list[str]
    weaknesses: list[str]
    hiring_recommendation: str
    recommendation_justification: str


class JDRequirement(BaseModel):
    requirement: str = Field(
        description="One atomic candidate requirement stated in the Job Description"
    )

    requirement_type: Literal[
        "required",
        "preferred"
    ] = Field(
        description="Whether the JD presents the requirement as required or preferred"
    )

    category: Literal[
        "skill",
        "tool",
        "experience",
        "education",
        "certification"
    ] = Field(
        description="The type of candidate requirement"
    )


class JDRequirements(BaseModel):
    requirements: list[JDRequirement] = Field(
        description="Atomic required and preferred candidate requirements extracted from the Job Description"
    )


class RequirementEvidenceAssessment(BaseModel):

    requirement: str = Field(
        description="The JD requirement, copied exactly as supplied."
    )

    requirement_type: Literal["required", "preferred"] = Field(
        description="Copied exactly as supplied."
    )

    category: Literal[
        "skill",
        "tool",
        "experience",
        "education",
        "certification"
    ] = Field(
        description="Copied exactly as supplied."
    )

    core_capability: str = Field(
        description=(
            "The capability, tool, qualification or experience the "
            "requirement names, with level words, thresholds and "
            "subjective descriptors removed."
        )
    )

    resume_term: str = Field(
        description=(
            "Search the whole resume context first. If the resume uses "
            "the core capability's own name anywhere, resume_term must be "
            "that exact word or phrase. Otherwise use the closest phrase "
            "the resume contains. Use 'none' if the resume contains "
            "nothing relevant."
        )
    )

    capability_match: Literal[
        "same",
        "related_only",
        "aspiration_only",
        "absent"
    ] = Field(
        description=(
            "'same' if resume_term names the core capability itself, "
            "names one of the alternatives or examples listed in the "
            "requirement, or describes the candidate doing that same thing. "
            "Naming it as something the candidate applies, uses or is "
            "skilled in is enough. Do not require the resume to state a "
            "proficiency level or to elaborate. "
            "'related_only' if resume_term is a different skill, even one "
            "that normally relies on the core capability. "
            "'aspiration_only' if the resume only expresses interest or a "
            "wish to gain it. "
            "'absent' if resume_term is 'none'."
        )
    )

    qualifier: str = Field(
        description=(
            "The level word, numeric threshold, named field or concrete "
            "context the requirement adds to the core capability. Use "
            "'none' if there is none. General wording such as proficiency, "
            "experience or ability, and subjective descriptors such as "
            "dynamic or effectively, are not qualifiers."
        )
    )

    qualifier_kind: Literal[
        "none",
        "elevated_level",
        "numeric_threshold",
        "named_item",
        "concrete_context"
    ] = Field(
        description="The kind of qualifier identified above."
    )

    qualifier_evidence: str = Field(
        description=(
            "The exact resume quote that satisfies the qualifier, or "
            "'none'. For an elevated level, the quote itself must state "
            "that level for this capability or show elevated scope, "
            "complexity or responsibility in it. A quote that only shows "
            "the capability being used is not qualifier evidence."
        )
    )

    qualifier_met: bool = Field(
        description=(
            "True only if qualifier_evidence is not 'none' and satisfies "
            "the qualifier. False if qualifier_kind is 'none'."
        )
    )

    supporting_evidence: list[str] = Field(
        description=(
            "Exact quotes copied from the resume context that justify the "
            "findings above. Empty only when nothing relevant exists."
        )
    )

    assessment_reason: str = Field(
        description=(
            "One or two sentences explaining the findings above."
        )
    )

    status: Literal["supported", "not_supported"] = Field(
        description="Overall result. This is recalculated in Python."
    )


class QualitativeEvaluation(BaseModel):
    candidate_summary: str
    recommendation_justification: str


jd_parser = PydanticOutputParser(pydantic_object=JDRequirements)
evidence_parser = PydanticOutputParser(pydantic_object=RequirementEvidenceAssessment)
qualitative_parser = PydanticOutputParser(pydantic_object=QualitativeEvaluation)

LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

llm = ChatOpenAI(model=LLM_MODEL, temperature=0) if API_KEY else None
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL) if API_KEY else None
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


jd_extraction_prompt = ChatPromptTemplate.from_template("""
You are extracting candidate requirements from a Job Description.

Extract ONLY requirements that can be used to evaluate whether a candidate
meets the Job Description.

Do NOT convert general responsibilities into candidate requirements.

ATOMICITY RULE

Each requirement must be independently scorable.

Ask:

"Could one meaningful part of this requirement be supported by a resume
while another meaningful part is absent?"

If YES, split it into separate requirements.

IMPORTANT:
When splitting a requirement, NEVER DROP any meaningful part of the
original requirement.

Example:

"Ability to work effectively in a dynamic, research-oriented group that
has several concurrent projects"

must produce BOTH:

- Ability to work effectively in a dynamic, research-oriented group
- Ability to work effectively on several concurrent projects

Another example:

"Proficiency with data mining, mathematics, and statistical analysis"

must produce:

- Proficiency with data mining
- Proficiency with mathematics
- Proficiency with statistical analysis

RULES

1. Preserve whether each requirement is REQUIRED or PREFERRED.

2. Split independent mandatory skills or capabilities.

3. Do NOT split alternatives or examples.

   Example:

   "Experience with programming languages (ex: Java/Python, SAS)"

   remains ONE requirement because the languages are examples or
   alternatives.

4. Preserve meaningful:
   - quantities
   - years
   - thresholds
   - qualification levels
   - degree requirements
   - certifications
   - words such as "advanced" or "research-oriented"

5. Do not weaken or strengthen the JD.

6. Do not invent requirements.

7. Remove duplicate requirements.

8. Assign the most appropriate category:
   - skill
   - tool
   - experience
   - education
   - certification

JOB DESCRIPTION:
{job_description}

{format_instructions}
""")


jd_audit_prompt = ChatPromptTemplate.from_template("""
You are auditing an extracted set of Job Description requirements.

Compare the ORIGINAL JOB DESCRIPTION with the DRAFT REQUIREMENTS.

Your job is to return the FINAL corrected requirement list.

AUDIT RULES

1. Every candidate qualification stated in the original JD must be represented.

2. No meaningful requirement may disappear when a compound requirement
   is split.

3. Each final requirement must be independently scorable.

Use this atomicity test:

"Could one meaningful part be supported by a resume while another
meaningful part is absent?"

If YES, split the requirement.

Example:

"Ability to work effectively in a dynamic, research-oriented group that
has several concurrent projects"

must result in BOTH:

- Ability to work effectively in a dynamic, research-oriented group
- Ability to work effectively on several concurrent projects

4. Split independent mandatory skills.

Example:

"Proficiency with data mining, mathematics, and statistical analysis"

must result in three independently scorable requirements.

5. Do NOT split alternatives or examples.

Example:

"Experience with programming languages (ex: Java/Python, SAS)"

remains one requirement.

6. Preserve all meaningful quantities, levels and qualifications including:
   - years of experience
   - "advanced"
   - degree requirements
   - certifications
   - "research-oriented"
   - other explicit thresholds

7. Preserve REQUIRED versus PREFERRED correctly.

8. Do not invent requirements that are not present in the original JD.

9. Do not convert general job responsibilities into candidate qualifications.

10. Remove duplicates.

ORIGINAL JOB DESCRIPTION:
{job_description}

DRAFT REQUIREMENTS:
{draft_requirements}

{format_instructions}
""")

evidence_prompt = ChatPromptTemplate.from_template("""
You are assessing ONE Job Description requirement against the resume of ONE candidate.

The JD requirement is the evaluation criterion. It is never evidence about the candidate.
Use ONLY the resume context below. Do not use outside knowledge.

HOW TO DECIDE

Step A - Find the capability.

Identify the core capability, tool, qualification or experience the requirement names.

It counts as present only if the resume:
- names it directly, or
- describes the candidate actually doing the same thing.

Different wording is acceptable when it clearly describes the same capability.

It does NOT count when:
- it is only implied by a different skill;
- doing X would normally rely on Y, but Y itself is not demonstrated;
- the resume only expresses interest, intent, aspiration, or a wish to gain exposure to it;
- the evidence is ambiguous.


Step B - Check each qualifier using the matching rule.

1. General capability wording
   Examples:
   - "proficiency with"
   - "experience with"
   - "ability to"
   - "knowledge of"

   These are satisfied when Step A is satisfied.

   The resume does NOT need to repeat words such as:
   - proficiency
   - experience
   - ability
   - knowledge

   Direct and unambiguous evidence of actually using or demonstrating the same capability
   is sufficient.


2. Elevated level wording
   Examples:
   - "advanced"
   - "expert"
   - "extensive"
   - "deep"

   These are satisfied only when the resume:
   - explicitly states that elevated level for the same capability, OR
   - provides direct and unambiguous evidence of an elevated level through the
     scope, complexity, responsibility, or depth of the work described.

   Repeated mentions or ordinary use alone do NOT prove an elevated level.

   If the elevated level is ambiguous, return "not_supported".


3. Numeric thresholds
   Examples:
   - years of experience
   - counts
   - minimum quantities

   These are satisfied only when:
   - a number stated in the resume, OR
   - employment dates shown in the resume

   meet the threshold for that specific kind of experience.

   State the number or duration you found in assessment_reason.

   Do not transfer years from one type of experience to another.


4. Named items
   Examples:
   - tools
   - programming languages
   - degrees
   - fields of study
   - certifications

   These must be named in the resume.

   When the requirement lists alternatives or examples, one of them is enough.

   When the JD explicitly allows "related" or "equivalent" items, accept only a
   resume qualification whose wording is clearly and directly related to the
   named field.

   If the relationship is ambiguous, return "not_supported".


5. Concrete context
   Examples:
   - a type of team
   - work setting
   - domain
   - responsibility
   - project environment

   The concrete context must be described in the resume.

   Different wording is acceptable when it clearly describes the same context.


6. Subjective descriptors
   Examples:
   - "dynamic"
   - "fast-paced"
   - "effectively"

   These are not separate factual qualifiers unless the JD makes them objectively
   measurable.

   Judge the concrete parts of the requirement instead.


Step C - Decide.

Return "supported" ONLY if:
- Step A is satisfied, AND
- every applicable qualifier rule above is satisfied.

Otherwise return "not_supported".

"not_supported" means only that the resume does not demonstrate the complete
requirement.

Do NOT claim that the candidate definitely lacks the capability.


EVIDENCE RULES

- supporting_evidence must contain exact quotes copied character-for-character
  from the supplied resume context.
- Do NOT paraphrase.
- Do NOT merge separate passages.
- Do NOT rewrite, clean, or improve the wording.
- Each quote must be one continuous passage of at most 30 words.
- For "supported":
  provide 1 to 3 quotes that prove the requirement.
- For "not_supported":
  provide the closest relevant quote if one exists, such as:
  - a stated number below a threshold;
  - a weaker or related capability;
  - aspirational wording.
- Use an empty list only when the resume contains no relevant evidence.


ASSESSMENT REASON

assessment_reason must:
- be one or two concise sentences;
- explain which rule determined the result;
- refer only to the JD requirement and supplied resume evidence;
- not introduce outside facts or assumptions.


METADATA

Copy these values exactly as supplied:
- requirement
- requirement_type
- category


JD REQUIREMENT:
{requirement}

REQUIREMENT TYPE:
{requirement_type}

CATEGORY:
{category}

RESUME CONTEXT:
{resume_context}

{format_instructions}
""")

qualitative_prompt = ChatPromptTemplate.from_template("""
Create the final qualitative evaluation for one candidate.

Use ONLY:
- the grounded requirement assessments,
- the deterministic match score,
- the fixed hiring recommendation below.

Do not recalculate the score.
Do not change the recommendation.
Do not invent candidate facts.
Unsupported means "not demonstrated in the resume".
Where an assessment reason states a specific fact, such as a number of
years, include that fact.

Generate only:
- candidate_summary
- recommendation_justification

candidate_summary:
- concise but informative;
- mention the candidate's strongest demonstrated areas;
- mention the most important unsupported requirements where relevant.

recommendation_justification:
- write 3 to 5 clear recruiter-ready sentences;
- explain why the candidate is or is not a fit;
- name the most important demonstrated strengths;
- name the most important unsupported REQUIRED requirements;
- mention preferred gaps separately where useful;
- be specific enough for HR to communicate an evidence-based application outcome;
- do not use vague blanket statements without naming the actual reasons.

CANDIDATE:
{candidate_name}

DETERMINISTIC MATCH SCORE:
{match_score}/100

FIXED HIRING RECOMMENDATION:
{hiring_recommendation}

GROUNDED REQUIREMENT ASSESSMENTS:
{assessment_context}

{format_instructions}
""")


jd_extraction_chain = jd_extraction_prompt | llm | jd_parser if llm else None
jd_audit_chain = jd_audit_prompt | llm | jd_parser if llm else None
evidence_chain = evidence_prompt | llm | evidence_parser if llm else None
qualitative_chain = qualitative_prompt | llm | qualitative_parser if llm else None


def infer_candidate_name(text, fallback):
    for line in [x.strip() for x in text.splitlines()[:8] if x.strip()]:
        words = line.split()
        alpha_ratio = sum(
            ch.isalpha() or ch.isspace() for ch in line
        ) / max(len(line), 1)

        if (
            2 <= len(words) <= 5
            and len(line) <= 60
            and "@" not in line
            and not any(ch.isdigit() for ch in line)
            and alpha_ratio > 0.85
        ):
            return line.title() if line.isupper() else line

    return fallback


def load_uploaded_pdf(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name

    try:
        documents = PyPDFLoader(temp_path).load()
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    return documents


def load_pdf_path(file_path):
    return PyPDFLoader(str(file_path)).load()


def documents_to_text(documents):
    return "\n".join(doc.page_content for doc in documents).strip()


@st.cache_data(show_spinner=False)
def extract_jd_requirements(job_description):
    draft = jd_extraction_chain.invoke({
        "job_description": job_description,
        "format_instructions": jd_parser.get_format_instructions()
    })

    final = jd_audit_chain.invoke({
        "job_description": job_description,
        "draft_requirements": draft.model_dump_json(indent=2),
        "format_instructions": jd_parser.get_format_instructions()
    })

    if not final.requirements:
        raise ValueError("No scorable candidate requirements could be extracted from the JD.")

    requirement_keys = [
        (
            item.requirement.strip().lower(),
            item.requirement_type,
            item.category
        )
        for item in final.requirements
    ]

    if len(requirement_keys) != len(set(requirement_keys)):
        raise ValueError("Duplicate JD requirements detected after audit.")

    return final


def recommendation_from_score(score):
    if score >= 80:
        return "Recommend for hire"
    if score >= 60:
        return "Consider for interview"
    return "Not recommended for hire"


def assess_requirement(item, retriever):
    """
    Assess one JD requirement against one candidate's resume.

    The LLM extracts structured evidence findings. Python then derives the
    final supported / not_supported status from those findings.
    """
    retrieved_docs = retriever.invoke(item.requirement)
    resume_context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    assessment = evidence_chain.invoke({
        "requirement": item.requirement,
        "requirement_type": item.requirement_type,
        "category": item.category,
        "resume_context": resume_context,
        "format_instructions":
            evidence_parser.get_format_instructions()
    })

    # Status is derived from the structured findings, not trusted
    # directly from the LLM.
    supported = (
        assessment.capability_match == "same"
        and (
            assessment.qualifier_kind == "none"
            or assessment.qualifier_met
        )
    )

    # JD requirement metadata remains authoritative.
    return assessment.model_copy(update={
        "requirement": item.requirement,
        "requirement_type": item.requirement_type,
        "category": item.category,
        "status": "supported" if supported else "not_supported"
    })


def assess_candidate(documents, candidate_name, jd_requirements):
    for doc in documents:
        doc.metadata["candidate_name"] = candidate_name

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("The resume could not be split into usable text chunks.")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": len(chunks)})

    assessments = [
        assess_requirement(item, retriever)
        for item in jd_requirements.requirements
    ]

    required = [
        a for a in assessments
        if a.requirement_type == "required"
    ]

    preferred = [
        a for a in assessments
        if a.requirement_type == "preferred"
    ]

    required_rate = (
        sum(a.status == "supported" for a in required) / len(required)
        if required else 0
    )

    preferred_rate = (
        sum(a.status == "supported" for a in preferred) / len(preferred)
        if preferred else 0
    )

    if required and preferred:
        score = required_rate * 80 + preferred_rate * 20
    elif required:
        score = required_rate * 100
    elif preferred:
        score = preferred_rate * 100
    else:
        score = 0

    score = round(score, 2)

    matching = [
        a.requirement
        for a in assessments
        if a.status == "supported"
    ]

    missing = [
        a.requirement
        for a in assessments
        if a.status == "not_supported"
    ]

    strengths = [
        (
            f"{a.requirement}: {a.supporting_evidence[0]}"
            if a.supporting_evidence
            else a.requirement
        )
        for a in assessments
        if a.status == "supported"
    ]

    weaknesses = [
        f"{a.requirement}: {a.assessment_reason}"
        for a in assessments
        if (
            a.status == "not_supported"
            and a.requirement_type == "required"
        )
    ]

    assessment_context = "\n\n".join(
        (
            f"REQUIREMENT: {a.requirement}\n"
            f"TYPE: {a.requirement_type.upper()}\n"
            f"STATUS: {a.status.upper()}\n"
            f"REASON: {a.assessment_reason}\n"
            f"EVIDENCE: "
            f"{'; '.join(a.supporting_evidence) if a.supporting_evidence else 'None'}"
        )
        for a in assessments
    )

    hiring_recommendation = recommendation_from_score(score)

    qualitative = qualitative_chain.invoke({
        "candidate_name": candidate_name,
        "match_score": score,
        "hiring_recommendation": hiring_recommendation,
        "assessment_context": assessment_context,
        "format_instructions":
            qualitative_parser.get_format_instructions()
    })

    evaluation = CandidateEvaluation(
        candidate_name=candidate_name,
        match_score=score,
        matching_skills=matching,
        missing_skills=missing,
        candidate_summary=qualitative.candidate_summary,
        strengths=strengths,
        weaknesses=weaknesses,
        hiring_recommendation=hiring_recommendation,
        recommendation_justification=
            qualitative.recommendation_justification
    )

    return evaluation, assessments


def evaluate_resume_documents(
    job_description,
    documents,
    candidate_name,
    source_name
):
    """Evaluate page-level PDF Documents without collapsing page structure."""
    jd_requirements = extract_jd_requirements(job_description)

    for doc in documents:
        doc.metadata["candidate_name"] = candidate_name
        doc.metadata["source_file"] = source_name

    evaluation, _ = assess_candidate(
        documents,
        candidate_name,
        jd_requirements
    )

    return evaluation


@st.cache_data(show_spinner=False)
def evaluate_resume_text(
    job_description,
    resume_text,
    candidate_name,
    source_name
):
    jd_requirements = extract_jd_requirements(job_description)

    documents = [
        Document(
            page_content=resume_text,
            metadata={
                "candidate_name": candidate_name,
                "source_file": source_name
            }
        )
    ]

    evaluation, _ = assess_candidate(
        documents,
        candidate_name,
        jd_requirements
    )

    return evaluation


def signature_for(jd, label, payload):
    digest = hashlib.sha256()
    digest.update(jd.strip().encode("utf-8"))
    digest.update(label.encode("utf-8"))
    digest.update(payload)
    return digest.hexdigest()


def render_result_card(title, items, positive=True):
    icon = "✓" if positive else "–"
    css_class = "tick" if positive else "gap"

    if items:
        rows = "".join(
            f'<div class="result-item">'
            f'<span class="{css_class}">{icon}</span>'
            f'<span>{html.escape(item)}</span>'
            f'</div>'
            for item in items
        )
    else:
        rows = (
            '<div class="result-item">'
            '<span>No items to display.</span>'
            '</div>'
        )

    st.markdown(
        f'<div class="result-card">'
        f'<h4>{html.escape(title)}</h4>'
        f'{rows}</div>',
        unsafe_allow_html=True
    )


def show_evaluation(result):
    score_col, identity_col = st.columns([1, 2], gap="large")

    with score_col:
        st.metric(
            "Match Score",
            f"{result.match_score:.2f}/100"
        )
        st.progress(
            max(0, min(100, int(round(result.match_score))))
        )

    with identity_col:
        st.markdown(
            f"## {html.escape(result.candidate_name)}"
        )
        st.markdown(
            f'<div class="recommendation-card">'
            f'<div class="recommendation-label">'
            f'Hiring Recommendation</div>'
            f'<div class="recommendation-value">'
            f'{html.escape(result.hiring_recommendation)}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("### Candidate Summary")
    st.write(result.candidate_summary)

    left, right = st.columns(2, gap="large")

    with left:
        render_result_card(
            "Matching Skills / Requirements",
            result.matching_skills,
            True
        )

    with right:
        render_result_card(
            "Missing / Not Demonstrated",
            result.missing_skills,
            False
        )

    left, right = st.columns(2, gap="large")

    with left:
        render_result_card(
            "Strengths",
            result.strengths,
            True
        )

    with right:
        render_result_card(
            "Weaknesses / Gaps",
            result.weaknesses,
            False
        )

    st.markdown("### Hiring Recommendation")
    st.markdown(
        f'<div class="recommendation-card">'
        f'<div class="recommendation-value">'
        f'{html.escape(result.hiring_recommendation)}</div>'
        f'<div class="recommendation-text">'
        f'{html.escape(result.recommendation_justification)}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


project_root = Path(__file__).resolve().parent
banner_path = project_root / "assets" / "Resume_analyzer.png"

if banner_path.exists():
    st.image(
        str(banner_path),
        use_container_width=True
    )
else:
    st.title("AI-powered Resume Screening Assistant")


with st.expander("How to use this tool"):
    st.markdown(
        """
**Steps**

1. Paste any Job Description below.
2. Upload one or more resume PDFs, then evaluate one candidate or compare several.

**How candidates are assessed**

- Candidate evidence comes only from the resume.
- Different wording counts when it clearly expresses the same capability.
- Related concepts are not automatically treated as equivalent.
- Qualifiers such as years of experience, an advanced level, degrees and certifications need explicit support in the resume.
- Absent or ambiguous evidence is treated as not demonstrated.
- Unstated or transferable competencies are not inferred.
"""
    )


job_description = st.text_area(
    "Paste Job Description",
    height=220,
    key="shared_jd",
    placeholder="Paste the complete job description here..."
)


if not API_KEY:
    st.warning(
        "OPENAI_API_KEY is not configured. "
        "Add it to a local .env file or your deployment "
        "environment/secrets before evaluating."
    )


tab1, tab2 = st.tabs([
    "Evaluate Candidate",
    "Compare Candidates"
])


with tab1:
    st.subheader("Evaluate Candidate")

    source = st.radio(
        "Resume source",
        [
            "Paste Resume Text",
            "Upload Resume PDF(s)",
            "Select Demo Candidate"
        ],
        horizontal=True
    )

    resume_text = ""
    candidate_name = ""
    source_label = ""
    payload = b""
    resume_documents = None

    if source == "Paste Resume Text":
        resume_text = st.text_area(
            "Paste Resume Text",
            height=240,
            placeholder="Paste the candidate's resume text here..."
        )

        if resume_text.strip():
            candidate_name = infer_candidate_name(
                resume_text,
                "Pasted Resume"
            )
            source_label = "pasted_resume"
            payload = resume_text.encode("utf-8")

    elif source == "Upload Resume PDF(s)":
        uploaded = st.file_uploader(
            "Upload Resume PDF(s)",
            type=["pdf"],
            accept_multiple_files=True,
            key="individual_uploads"
        )

        if uploaded:
            selected_name = st.selectbox(
                "Select Candidate to Evaluate",
                [file.name for file in uploaded]
            )

            selected_file = next(
                file
                for file in uploaded
                if file.name == selected_name
            )

            payload = selected_file.getvalue()
            source_label = selected_file.name

            try:
                docs = load_uploaded_pdf(selected_file)
                resume_documents = docs
                resume_text = documents_to_text(docs)

                if not resume_text:
                    raise ValueError(
                        "Usable resume text could not be extracted."
                    )

                candidate_name = infer_candidate_name(
                    resume_text,
                    Path(selected_file.name).stem
                )

            except Exception as exc:
                st.error(str(exc))
                resume_text = ""

    else:
        available = {
            path.stem.replace("_", " "): path
            for path in sorted((project_root / "data").glob("*.pdf"))
        }

        if available:
            demo_label = st.selectbox(
                "Select from Shortlisted Candidates",
                list(available)
            )

            demo_path = available[demo_label]
            payload = demo_path.read_bytes()
            source_label = demo_label

            try:
                docs = load_pdf_path(demo_path)
                resume_documents = docs
                resume_text = documents_to_text(docs)

                if not resume_text:
                    raise ValueError(
                        "Usable resume text could not be extracted."
                    )

                candidate_name = infer_candidate_name(
                    resume_text,
                    demo_label
                )

            except Exception as exc:
                st.error(str(exc))
                resume_text = ""

        else:
            st.info(
                "Demo resumes were not found in the project data folder."
            )

    current_signature = (
        signature_for(
            job_description,
            source_label,
            payload
        )
        if (
            job_description.strip()
            and resume_text.strip()
            and payload
        )
        else None
    )

    if (
        "individual_signature" in st.session_state
        and st.session_state.individual_signature
        != current_signature
    ):
        st.session_state.pop(
            "individual_result",
            None
        )
        st.session_state.pop(
            "individual_signature",
            None
        )

    if st.button(
        "Evaluate Resume",
        type="primary",
        key="evaluate_individual"
    ):
        if not API_KEY:
            st.error(
                "Please configure OPENAI_API_KEY before evaluating."
            )

        elif not job_description.strip():
            st.error(
                "Please enter a Job Description before evaluating candidates."
            )

        elif not resume_text.strip():
            st.error(
                "Please provide a resume."
            )

        else:
            try:
                with st.spinner(
                    "Evaluating candidate..."
                ):
                    if resume_documents is not None:
                        result = evaluate_resume_documents(
                            job_description,
                            resume_documents,
                            candidate_name or source_label,
                            source_label
                        )
                    else:
                        result = evaluate_resume_text(
                            job_description,
                            resume_text,
                            candidate_name or source_label,
                            source_label
                        )

                st.session_state.individual_result = result
                st.session_state.individual_signature = (
                    current_signature
                )

            except Exception:
                st.error(
                    "The candidate evaluation could not be completed. "
                    "Please verify the resume, API configuration, "
                    "and try again."
                )

    if (
        st.session_state.get("individual_result")
        is not None
        and st.session_state.get("individual_signature")
        == current_signature
    ):
        show_evaluation(
            st.session_state.individual_result
        )


with tab2:
    st.subheader("Compare Candidates")

    compare_uploads = st.file_uploader(
        "Upload at least two Resume PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="comparison_uploads"
    )

    compare_payload = (
        b"".join(
            file.name.encode("utf-8")
            + file.getvalue()
            for file in compare_uploads
        )
        if compare_uploads
        else b""
    )

    compare_signature = (
        signature_for(
            job_description,
            "comparison",
            compare_payload
        )
        if (
            job_description.strip()
            and len(compare_uploads) >= 2
        )
        else None
    )

    if (
        "comparison_signature" in st.session_state
        and st.session_state.comparison_signature
        != compare_signature
    ):
        st.session_state.pop(
            "comparison_results",
            None
        )
        st.session_state.pop(
            "comparison_signature",
            None
        )

    if st.button(
        "Compare Candidates",
        type="primary",
        key="compare_candidates"
    ):
        if not API_KEY:
            st.error(
                "Please configure OPENAI_API_KEY before evaluating."
            )

        elif not job_description.strip():
            st.error(
                "Please enter a Job Description before evaluating candidates."
            )

        elif len(compare_uploads) < 2:
            st.error(
                "Please upload at least two resumes for comparison."
            )

        else:
            try:
                with st.spinner(
                    "Evaluating candidates independently..."
                ):
                    results = []
                    failures = []

                    for uploaded_file in compare_uploads:
                        try:
                            docs = load_uploaded_pdf(
                                uploaded_file
                            )

                            resume_text = documents_to_text(
                                docs
                            )

                            if not resume_text:
                                raise ValueError(
                                    "No usable resume text extracted."
                                )

                            candidate_name = infer_candidate_name(
                                resume_text,
                                Path(
                                    uploaded_file.name
                                ).stem
                            )

                            result = evaluate_resume_documents(
                                job_description,
                                docs,
                                candidate_name,
                                uploaded_file.name
                            )

                            results.append(
                                result
                            )

                        except Exception as exc:
                            failures.append(
                                (
                                    uploaded_file.name,
                                    str(exc)
                                )
                            )

                    if failures:
                        failed_names = ", ".join(
                            name
                            for name, _ in failures
                        )

                        st.warning(
                            "Could not evaluate: "
                            + failed_names
                        )

                    if len(results) >= 2:
                        results.sort(
                            key=lambda x:
                                x.match_score,
                            reverse=True
                        )

                        st.session_state.comparison_results = (
                            results
                        )

                        st.session_state.comparison_signature = (
                            compare_signature
                        )

                    else:
                        st.error(
                            f"Only {len(results)} of "
                            f"{len(compare_uploads)} uploaded resumes "
                            "were evaluated successfully. "
                            "At least 2 successful evaluations are "
                            "required for comparison."
                        )

            except Exception:
                st.error(
                    "Candidate comparison could not be completed. "
                    "Please verify the files, API configuration, "
                    "and try again."
                )

    results = st.session_state.get(
        "comparison_results"
    )

    if (
        results
        and st.session_state.get(
            "comparison_signature"
        )
        == compare_signature
    ):
        rows = []

        for rank, result in enumerate(
            results,
            start=1
        ):
            rows.append({
                "Rank": rank,
                "Candidate":
                    result.candidate_name,
                "Match Score":
                    result.match_score,
                "Matching":
                    len(result.matching_skills),
                "Missing":
                    len(result.missing_skills),
                "Recommendation":
                    result.hiring_recommendation
            })

        comparison_df = pd.DataFrame(
            rows
        )

        st.markdown(
            "### Candidate Ranking"
        )

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        best = results[0]

        st.markdown(
            "### Highest-Ranked Candidate"
        )

        best_col1, best_col2 = (
            st.columns(
                [2, 1],
                gap="large"
            )
        )

        with best_col1:
            st.markdown(
                f'<div class="best-fit-card">'
                f'<div class="recommendation-label">'
                f'Highest-Ranked Candidate</div>'
                f'<div class="recommendation-value">'
                f'{html.escape(best.candidate_name)}</div>'
                f'<div class="recommendation-text">'
                f'{html.escape(best.recommendation_justification)}'
                f'</div></div>',
                unsafe_allow_html=True
            )

        with best_col2:
            st.metric(
                "Top Match Score",
                f"{best.match_score:.2f}/100"
            )

        st.markdown(
            "### Why this candidate ranks first"
        )

        st.write(
            f"{best.candidate_name} has the highest "
            f"deterministic match score "
            f"({best.match_score:.2f}/100), with "
            f"{len(best.matching_skills)} supported "
            f"requirements and "
            f"{len(best.missing_skills)} requirements "
            f"not demonstrated."
        )


st.divider()

st.caption(
    f"**Model:** OpenAI `{LLM_MODEL}` for requirement extraction, "
    f"assessment and summaries; `{EMBEDDING_MODEL}` for resume retrieval."
)

st.caption(
    f"**Assessed limitation with {LLM_MODEL}:** `{LLM_MODEL}` may "
    "occasionally misclassify requirements involving numeric thresholds or "
    "closely related experience wording, even when relevant evidence is "
    "present in the resume. These cases are conservatively counted as not "
    "demonstrated. For example, \"9+ years\" against \"seven or more "
    "years\" may be shown as not demonstrated. For required requirements, "
    "the model's reason is shown beside the item under Weaknesses / Gaps."
)
