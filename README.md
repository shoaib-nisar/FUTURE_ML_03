Resume Screening & Candidate Ranking System

A lightweight NLP pipeline that ranks resumes against a job description using a blend of semantic similarity (TF-IDF + cosine similarity) and weighted skill matching (spaCy phrase matching against a curated skills taxonomy).

Given a job description and a pool of resumes, it returns a ranked table of candidates with a transparent, explainable score for each one — no black-box model, just two interpretable signals combined.


How It Works


Clean the text. Resumes and the job description are lowercased, stripped of punctuation/numbers, and stopwords are removed.
Extract skills. A MASTER_SKILLS list (~60 terms spanning IT, Sales, HR, Design, Finance, and soft skills) is matched against both the job description and every resume using a spaCy PhraseMatcher. This produces two skill sets: required_skills (from the JD) and Extracted_Skills (per candidate).
Weight the requirements. Any skill passed in as must_have_skills gets a weight of 2.0; every other skill mentioned in the JD gets a weight of 1.0.
Score each candidate on two independent signals, then blend them (see below).
Rank and explain. Candidates are sorted by Final_Score, and a short natural-language verdict is generated per candidate (strong / partial / weak match), listing matched and missing skills.



The Scoring Formula

Each candidate gets two sub-scores that are combined into one Final_Score:

1. TF-IDF Similarity (default weight: 0.6)

The job description and every resume are vectorized together with TfidfVectorizer, and the cosine similarity between the JD vector and each resume vector is computed. This captures overall textual/topical closeness — how much a resume "sounds like" the job description, beyond just keyword hits.

2. Weighted Skill Overlap (default weight: 0.4)

Skill_Overlap = (sum of weights for matched skills) / (sum of weights for all required skills)

Only skills that appear in both the JD's required-skill set and the candidate's extracted-skill set count. Must-have skills contribute 2x as much to the numerator (and denominator) as nice-to-haves, so missing a must-have skill hurts more than missing an optional one.

3. Final Score

Final_Score = (0.6 × TF-IDF_Similarity) + (0.4 × Skill_Overlap)

The 0.6 / 0.4 split is configurable via sim_weight / skill_weight in screen_resumes(). The blend exists because either signal alone is misleading: TF-IDF alone rewards resumes that are wordy/topically dense but skill-light; raw skill overlap alone ignores everything else in a strong resume (impact, scope, tenure) that TF-IDF partially captures through shared vocabulary.

Verdict Bands

Final ScoreVerdict≥ 0.40Strong match0.20 – 0.39Partial match< 0.20Weak match


Worked Example: IT Role

Job description:


"We are looking for an IT professional with strong skills in Python, SQL, networking, cloud platforms (AWS or Azure), Linux, cybersecurity, and troubleshooting. Experience with system administration and database management is required. Communication and problem solving skills are essential."



Must-have skills specified: python, sql, networking (weight 2.0 each)

All other extracted required skills (weight 1.0 each): azure, aws, linux, cloud, cybersecurity, troubleshooting, system administration, database management, communication, problem solving

That's 13 required skills total. Total weight = (3 × 2.0) + (10 × 1.0) = 16.0

Candidate #1 — ID 20824105 (ranked 1st out of 120 IT resumes)

Matched skills: networking, linux, python, aws, communication, sql, cloud, problem solving

Weight of matched skills = 2.0 (python) + 2.0 (sql) + 2.0 (networking) + 1.0×5 (linux, aws, communication, cloud, problem solving) = 11.0

Skill_Overlap = 11.0 / 16.0 = 0.5000

TF-IDF Similarity (from cosine similarity against the JD vector) = 0.1770

Final_Score = (0.6 × 0.1770) + (0.4 × 0.5000)
            =    0.1062      +    0.2000
            =    0.3062

→ Verdict: Partial match (0.31) — strong on required skills (half of the weighted requirement met, including all three must-haves) but the resume's overall language doesn't overlap heavily with the JD's phrasing, which is what keeps this out of "strong match" territory.

Full Top-5, IT Role

RankIDTF-IDF SimSkill OverlapFinal ScoreVerdict1208241050.17700.50000.3062Partial2462602300.09490.56250.2819Partial3100894340.09330.50000.2560Partial4176887660.08180.50000.2491Partial5180675560.07770.50000.2466Partial

Notice rank #2 has a lower TF-IDF similarity than #1 but a higher skill overlap (0.5625, driven by matching all three must-haves plus more nice-to-haves) — yet still ranks below #1. That's the 0.6/0.4 blend at work: TF-IDF is weighted more heavily, so a candidate needs both signals working together to climb, not just a high skill count.


Usage

pythonranking, required_skills = screen_resumes(
    resume_df,
    job_description,
    must_have_skills=["python", "sql", "networking"],
    sim_weight=0.6,
    skill_weight=0.4,
)

print_summary(ranking, top_n=5)   # readable per-candidate verdicts
plot_ranking(ranking, top_n=10)   # horizontal bar chart of top scores

Requirements

pandas, numpy, matplotlib
nltk (stopwords corpus)
spacy (en_core_web_sm model)
scikit-learn (TfidfVectorizer, cosine_similarity)

Dataset

Expects a Resume.csv with columns ID, Resume_str, Resume_html, Category (this project uses the public Kaggle Resume dataset, 2,484 resumes across 24 job categories).

Known Limitations


Skill extraction is exact-phrase matching, not semantic — a resume that says "orchestrated containers" instead of "docker" or "kubernetes" won't get credit.
MASTER_SKILLS is hand-curated and finite. Anything outside that list (e.g. a specific programming framework) is invisible to the skill-overlap score, though it can still lift TF-IDF similarity.
TF-IDF has no notion of seniority, recency, or context — it only measures shared vocabulary, so a resume that lists relevant keywords in an unrelated context can still score reasonably well.
