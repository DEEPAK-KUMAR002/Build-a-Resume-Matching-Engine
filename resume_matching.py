import math

# ─────────────────────────────────────────────────────────────────────────────
# REDROB AI CAMPUS HACKATHON
# RESUME MATCHING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

# ─── SKILL_ALIASES (EXACT FROM PROBLEM SHEET) ───────────────────────────────
SKILL_ALIASES = {

    # Languages
    "python": "python",
    "pyhton": "python",
    "java": "java",

    "javascript": "javascript",
    "javascrpit": "javascript",
    "js": "javascript",

    "typescript": "typescript",
    "typescrpit": "typescript",

    "c++": "cpp",
    "cpp": "cpp",

    "r": "r",
    "kotlin": "kotlin",

    # ML / Data
    "machinelearning": "machine_learning",
    "machine learning": "machine_learning",
    "ml": "machine_learning",
    "sklearn": "machine_learning",

    "deeplearning": "deep_learning",
    "deep learning": "deep_learning",
    "deep-learning": "deep_learning",

    "tensorflow": "tensorflow",
    "pytorch": "pytorch",
    "keras": "keras",

    "nlp": "nlp",
    "bert": "bert",
    "xgboost": "xgboost",

    "feature engineering": "feature_engineering",

    "statistics": "statistics",
    "stats": "statistics",

    "regression": "regression",
    "clustering": "clustering",

    "data-viz": "data_visualization",
    "data visualization": "data_visualization",
    "data viz": "data_visualization",
    "matplotlib": "data_visualization",
    "tableau": "data_visualization",
    "power-bi": "data_visualization",
    "power bi": "data_visualization",
    "powerbi": "data_visualization",

    "pandas": "pandas",
    "numpy": "numpy",

    # Frontend
    "react": "react",
    "reacts": "react",
    "reactjs": "react",

    "vue": "vue",
    "vue.js": "vue",
    "vuejs": "vue",

    "redux": "redux",
    "tailwind": "tailwind",

    "html/css": "html_css",
    "html css": "html_css",
    "html": "html_css",
    "css": "html_css",

    "jest": "jest",
    "graphql": "graphql",

    # Backend
    "node.js": "nodejs",
    "nodejs": "nodejs",
    "node js": "nodejs",

    "flask": "flask",

    "spring boot": "spring_boot",
    "springboot": "spring_boot",

    "rest api": "rest_api",
    "rest": "rest_api",
    "restapi": "rest_api",

    "microservices": "microservices",

    # Databases
    "sql": "sql",
    "mysql": "mysql",
    "mysq": "mysql",

    "postgresql": "postgresql",
    "postgres": "postgresql",

    "mongodb": "mongodb",
    "redis": "redis",

    # DevOps / Cloud
    "docker": "docker",

    "kubernetes": "kubernetes",
    "kubernates": "kubernetes",
    "k8s": "kubernetes",

    "ci/cd": "ci_cd",
    "cicd": "ci_cd",
    "ci cd": "ci_cd",

    "aws": "aws",

    # Mobile
    "android": "android",
    "firebase": "firebase",

    # CS Fundamentals
    "algorithms": "algorithms",
    "algoritms": "algorithms",

    "data structure": "data_structures",
    "data structures": "data_structures",

    "competitive programming": "competitive_programming",

    # Design
    "ui/ux": "ui_ux",
    "ui ux": "ui_ux",
    "figma": "figma",
}

# ─── MULTI WORD MATCHING ────────────────────────────────────────────────────
multi_word_keys = sorted(
    [k for k in SKILL_ALIASES if " " in k],
    key=lambda x: -len(x)
)

# ─── SKILL NORMALIZATION ────────────────────────────────────────────────────
def normalize_skills(raw):

    tokens = [t.strip().lower() for t in raw.split(",")]

    canonical = []
    seen = set()

    for token in tokens:

        matched = None

        # Multi-word first
        for mwk in multi_word_keys:
            if token == mwk:
                matched = SKILL_ALIASES[mwk]
                break

        # Single token
        if matched is None and token in SKILL_ALIASES:
            matched = SKILL_ALIASES[token]

        # Deduplicate
        if matched is not None and matched not in seen:
            seen.add(matched)
            canonical.append(matched)

    return canonical

# ─── RESUME DATA ────────────────────────────────────────────────────────────
resumes_raw = [

    ("Arjun Sharma",
     "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning"),

    ("Priya Nair",
     "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS"),

    ("Rahul Gupta",
     "Java, Spring Boot, MySql, Microservices, Docker, kubernates"),

    ("Sneha Patel",
     "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib"),

    ("Vikram Singh",
     "C++, Algoritms, Data Structure, competitive programming, python"),

    ("Ananya Krishnan",
     "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD"),

    ("Karan Mehta",
     "Python, Sklearn, XGboost, feature engineering, SQL, tableau"),

    ("Deepika Rao",
     "Java, Android, Kotlin, Firebase, REST, UI/UX, figma"),

    ("Aditya Kumar",
     "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest"),

    ("Meera Iyer",
     "python, R, statistics, ML, regression, clustering, Power-BI"),
]

# ─── JOB DESCRIPTIONS ───────────────────────────────────────────────────────
jd_raw = {

    "JD-1 — Kakao (ML Engineer)":
        "Python, Machine Learning, Deep Learning, TensorFlow, "
        "PyTorch, SQL, Data Visualization, NLP, BERT, "
        "Feature Engineering, Statistics",

    "JD-2 — Naver (Backend Engineer)":
        "Java, Spring Boot, MySQL, PostgreSQL, Microservices, "
        "Docker, Kubernetes, REST API, CI/CD, Redis",

    "JD-3 — Line (Frontend Engineer)":
        "JavaScript, React, Vue, TypeScript, REST API, "
        "HTML/CSS, Node.js, GraphQL, Redux, Jest, AWS",
}

# ─── STEP 1 & 2 : NORMALIZATION + DEDUPLICATION ────────────────────────────
resumes = [
    (name, normalize_skills(raw))
    for name, raw in resumes_raw
]

# ─── STEP 3 : VOCABULARY CONSTRUCTION ───────────────────────────────────────
vocab = sorted(
    set(skill for _, skills in resumes for skill in skills)
)

vocab_index = {
    skill: i
    for i, skill in enumerate(vocab)
}

V = len(vocab)
N_DOCS = len(resumes)

# ─── STEP 4 : TF-IDF CALCULATION ────────────────────────────────────────────

# Document Frequency
df = {
    skill: sum(1 for _, skills in resumes if skill in skills)
    for skill in vocab
}

# IDF
idf = {
    skill: math.log(N_DOCS / df[skill])
    for skill in vocab
}

# TF-IDF vectors
tfidf_vectors = []

for name, skills in resumes:

    vec = [0.0] * V

    N = len(skills)

    for skill in skills:

        tf = 1 / N
        tfidf = tf * idf[skill]

        vec[vocab_index[skill]] = tfidf

    tfidf_vectors.append(vec)

# ─── STEP 5 : JD BINARY VECTORS ─────────────────────────────────────────────
jd_vectors = {}

for jd_name, raw in jd_raw.items():

    skills = normalize_skills(raw)

    vec = [0.0] * V

    for skill in skills:

        if skill in vocab_index:
            vec[vocab_index[skill]] = 1.0

    jd_vectors[jd_name] = vec

# ─── STEP 6 : COSINE SIMILARITY ─────────────────────────────────────────────
def cosine_similarity(a, b):

    dot = sum(x * y for x, y in zip(a, b))

    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)

# ─────────────────────────────────────────────────────────────────────────────
# FINAL CUSTOMIZED OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("                REDROB AI CAMPUS HACKATHON")
print("                    RESUME MATCHING ENGINE")
print("=" * 72)

for jd_name, jd_vec in jd_vectors.items():

    print(f"\n{jd_name}")
    print("-" * 72)

    scores = []

    for i, (candidate_name, skills) in enumerate(resumes):

        score = cosine_similarity(tfidf_vectors[i], jd_vec)

        scores.append((candidate_name, score, skills))

    # Sort by score desc, then alphabetical
    scores.sort(key=lambda x: (-x[1], x[0]))

    top3 = scores[:3]

    for rank, (name, score, skills) in enumerate(top3, start=1):

        matched_skills = []

        jd_skills = normalize_skills(jd_raw[jd_name])

        for skill in skills:
            if skill in jd_skills:
                matched_skills.append(skill)

        print(f"\n🏆 Rank #{rank}")
        print(f"Candidate      : {name}")
        print(f"Match Score    : {score:.2f}")

        print("Matched Skills : ", end="")
        print(", ".join(matched_skills))

        # Match level
        if score >= 0.50:
            level = "Excellent Match"
        elif score >= 0.30:
            level = "Good Match"
        else:
            level = "Average Match"

        print(f"Status         : {level}")

print("\n" + "=" * 72)
print("                    MATCHING COMPLETED")
print("=" * 72)

# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL VALIDATION OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

print("\n\nVALIDATION SECTION")
print("=" * 72)

print("\nNormalized Skills\n")

for name, skills in resumes:
    print(f"{name:20s} -> {skills}")

print("\nVocabulary\n")
print(vocab)

print("\nDocument Frequency + IDF\n")

for skill in vocab:
    print(
        f"{skill:25s} "
        f"DF={df[skill]} "
        f"IDF={idf[skill]:.4f}"
    )