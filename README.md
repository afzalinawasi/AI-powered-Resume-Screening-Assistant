# AI Resume Screening Assistant

An evidence-grounded **AI Resume Screening Assistant** built using **LangChain, RAG, OpenAI, FAISS, Pydantic, and Streamlit** to evaluate resumes against a Job Description, identify demonstrated and missing requirements, generate structured candidate assessments, and rank multiple candidates using transparent scoring.


---

## Project Overview & Streamlit UI 

The project evaluates one or more resumes against a supplied Job Description while keeping each candidate's evidence isolated.

The system produces:

- Match Score from 0–100
- Matching / demonstrated requirements
- Missing / not demonstrated requirements
- Candidate summary
- Evidence-backed strengths
- Required gaps / weaknesses
- Hiring recommendation
- Recommendation justification
- Multi-candidate ranking

The workflow is deliberately **evidence-grounded**: candidate facts must come from the supplied resume, and missing or ambiguous evidence is treated as **not demonstrated**, rather than inferred.

### Using the Streamlit App

**Streamlit App UI link**: [Click here](https://ai-powered-resume-screening-assistant.streamlit.app/)

1. **Paste the Job Description** e.g. a Data scientist job description in the input box at the top of the application.
2. Choose one of the two workflows:

### 1. Evaluate Candidate
Select a resume source:
- A. Paste Resume Text, or 
- B. Upload Resume PDF(s), or 
- C Select Demo Candidate

Then click **Evaluate Resume**.

The application displays the candidate's Match Score, hiring recommendation, summary, matching and missing requirements, strengths, weaknesses, and recommendation justification.


### 2. Compare Candidates
Upload **at least two resume PDFs** and click **Compare Candidates**.

The application evaluates each resume independently against the same Job Description and displays the candidate ranking, scores, supported/missing requirements, and highest-ranked candidate.

> **Please note:** Evaluation may take approximately **2–4 minutes per resume** because the current prototype has not yet been optimized for speed. Please allow the evaluation to complete before submitting another request.


---

## Key features

- PDF resume loading with `PyPDFLoader`
- Resume chunking with `RecursiveCharacterTextSplitter`
- OpenAI embeddings with `text-embedding-3-small`
- Separate FAISS vector store and retriever for each candidate
- Two-pass Job Description requirement extraction and audit
- Atomic required / preferred requirement classification
- Requirement-by-requirement resume evidence assessment
- Structured Pydantic outputs
- Deterministic Match Score calculation
- Deterministic hiring recommendation
- Evidence-backed strengths and required gaps
- Single-candidate evaluation
- Multiple-candidate comparison and ranking
- Audit trail for unsupported requirements
- Streamlit recruiter-facing interface

---

## Technical stack

| Component | Implementation |
|---|---|
| LLM | OpenAI `gpt-4o-mini` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Orchestration | LangChain |
| PDF Loading | `PyPDFLoader` |
| Text Splitting | `RecursiveCharacterTextSplitter` |
| Vector Store | FAISS |
| Structured Output | Pydantic + LangChain parsers |
| UI | Streamlit |
| Supporting Libraries | Pandas, python-dotenv |

---

## How it works

```text
Job Description
      |
      v
Requirement Extraction
      |
      v
Requirement Audit
      |
      v
Required / Preferred Atomic Requirements
      |
      v
Candidate Resume
      |
      v
PDF Loading -> Chunking -> Embeddings
      |
      v
Candidate-Specific FAISS Retriever
      |
      v
Requirement-by-Requirement Evidence Assessment
      |
      v
Supported / Not Supported
      |
      +-----------------------+
      |                       |
      v                       v
Deterministic Score      Grounded Summary
      |                   & Justification
      +-----------+-----------+
                  |
                  v
        Final Candidate Report
                  |
                  v
        Comparison & Ranking
```

Each candidate is processed independently so evidence from one resume cannot be used in another candidate's evaluation.

---

## Evidence policy

The project follows a conservative evidence policy:

- Candidate facts must come from the resume. This is strictly mentioned in the assignment. I have not built the logic to be purely inferential. 
- Different wording is acceptable only when it clearly represents the same capability.
- Related concepts are not automatically treated as equivalent.
- Aspirational statements such as *seeking exposure to* a skill are not treated as demonstrated experience.
- Experience duration, tools, qualifications, certifications, and work context must be demonstrated where required.
- Elevated requirements such as **advanced**, **expert**, **extensive**, or **deep** require evidence of that elevated level.
- Ambiguous or missing evidence is classified as **not demonstrated**.

**Note: This project is intentionally not a general-purpose engine for inferring unstated or transferable competencies.**

---

## Scoring

The LLM does **not** generate the Match Score.

When both required and preferred requirements exist:

```text
Match Score =
(Required Supported / Total Required × 80)
+
(Preferred Supported / Total Preferred × 20)
```

- Required requirements: **80%**
- Preferred requirements: **20%**

The same rule is applied to every candidate.

### Hiring recommendation

```text
Score >= 80       -> Recommend for hire
Score >= 60       -> Consider for interview
Score < 60        -> Not recommended for hire
```

The recommendation is calculated in Python. For the final candidate report, the qualitative LLM call generates only the candidate summary and recommendation justification. The Match Score, requirement statuses, strengths, weaknesses, and hiring recommendation are derived deterministically in Python from the structured assessments.

---

## Validation

The final notebook includes validation tests for:

1. Individual candidate evaluation
2. Candidate-to-candidate comparison
3. Missing requirement identification
4. Best-candidate selection
5. Hiring recommendation and justification

Targeted regression checks also verify important evidence-grounding behavior, including distinguishing demonstrated experience from aspirational wording.

---

## Performance note

During current testing, a single-candidate evaluation typically takes approximately **2–4 minutes**.

Runtime varies with:

- number of extracted JD requirements;
- resume length;
- network conditions; and
- OpenAI API latency.

The current prototype prioritizes **evidence-grounded evaluation and explainability over inference speed** and has not yet been optimized for latency.

Multi-candidate comparison may take proportionally longer.

---

## Known limitations

The current implementation uses `gpt-4o-mini` for JD requirement extraction/audit, structured evidence assessment, and final qualitative summaries.

Despite conservative prompts and structured outputs, the model may occasionally misclassify:

- numeric thresholds;
- closely related wording;
- semantic equivalents;
- elevated experience levels; or
- contextual requirements.

This can produce both:

- **false negatives** — valid resume evidence is not credited; and
- **false positives** — related evidence is incorrectly promoted to full support.

These classification errors can therefore affect the Match Score in either direction. I have evaluated this with multiple resumes and the results have been inconsistent. Perhaps a higher model will better computing capability can be used in future enhancements. 

The application should be used as a **screening assistant** project, not as a hiring assessment tool. 

---

## API Key & Environment configuration

For security, the `.env` file and `OPENAI_API_KEY` are not included in this GitHub repository.

If the project is executed locally, create a `.env` file and add a valid OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key
```


Submission note: For grading purposes, an OpenAI API key has been securely configured in Streamlit Secrets for the deployed application. It will remain available temporarily during the grading period and will be removed after the project has been evaluated.


---

## Future enhancements

- Evaluate higher-capability LLMs for improved evidence classification accuracy. 
- Handle numeric thresholds deterministically in Python
- Parallelize or batch independent requirement assessments
- Add caching and reduce repeated LLM context
- Add a visible progress loader during candidate evaluation
- Tune and benchmark FAISS retrieval strategy
- Add reranking or hybrid semantic + keyword retrieval
- Build a labelled evaluation dataset for precision / recall testing
- Extend the solution into a more generic cross-domain resume analyzer
- Improve handling of transferable skills and semantic equivalents
- Add exportable recruiter reports and comparison summaries
- Add production features such as authentication, monitoring, retries, and secure persistence

---

## Responsible use

This is an educational AI prototype project intended to **assist human resume review**.
LLM-generated assessments can contain errors or omissions. Recruiters should review the original resume, Job Description, and supporting evidence before making employment decisions.

**Final hiring decisions should remain with a qualified human reviewer.**

