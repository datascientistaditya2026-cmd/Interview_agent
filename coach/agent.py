import re
import random
from collections import Counter
from .documents import chunk_document

# ---------------- Settings ----------------

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_embedder = None                       # loaded once, then reused

STOP = set("""this that with from have were been will would could should their there which about into than
then them they what when where while your also such these those each other only over more most some very
just like because using used use can may might does did done being are was for and the not but you our
out its has had how why who""".split())

# Words that must not appear in the subject of a question ("What can How do?")
SKIP_SUBJECTS = {"it", "this", "that", "they", "there", "which", "these", "those", "he", "she", "we", "you", "i",
                 "how", "what", "why", "when", "where", "who", "if", "so", "but", "and", "or", "in", "on", "for",
                 "to", "as", "also", "however", "then", "here", "with", "by", "at", "of", "from", "while",
                 "because", "since", "although", "thus", "therefore", "not", "never", "often", "usually"}

# Subjects that are too generic to make a good question ("What can the user do?")
GENERIC = {"user", "users", "developer", "developers", "programmer", "programmers", "person", "people",
           "one", "someone", "everyone", "anyone", "thing", "things", "way", "ways", "example", "examples"}

DONT_KNOW = {"", "i don't know", "i dont know", "dont know", "don't know", "idk", "no idea"}

# Question templates. Each pattern matches a sentence in the document and gives a question about it.
SUBJ = r"(?P<s>(?:An?\s+|The\s+)?[^,.:;()]{2,40}?)"
PATTERNS = [
    (re.compile(rf"^{SUBJ}\s+(?P<v>is|are)\s+(?:mainly\s+|commonly\s+|often\s+)?used\s+(?:for|to|in)\b", re.I),
     "What {v} {s} used for?"),
    (re.compile(rf"^{SUBJ}\s+(?P<v>is|are)\s+(?:an?|the|one of|a type of|a kind of)\b", re.I),
     "What {v} {s}?"),
    (re.compile(rf"^{SUBJ}\s+(?:allows|enables|lets|helps|provides|supports|offers|makes|creates|returns|performs)\b", re.I),
     "What does {s} do?"),
    (re.compile(rf"^{SUBJ}\s+(?:can|may)\s+\w+", re.I),
     "What can {s} do?"),
]

CLEAN_SUBJECT = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _\-\+#]*")   # normal characters only


def get_embedder():
    """Load the embedding model on first use only. It turns text into numbers that capture meaning."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMB_MODEL)
    return _embedder


# ---------------- Text helpers ----------------

REPLACE = {"，": ",", "。": ".", "：": ":", "；": ";", "（": "(", "）": ")", "！": "!", "？": "?",
           "“": '"', "”": '"', "‘": "'", "’": "'", "\u3000": " ", "\xa0": " "}


def normalize(text: str) -> str:
    """Remove invisible characters and turn full-width punctuation into normal punctuation."""
    text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff\u00ad]", "", text)
    for bad, good in REPLACE.items():
        text = text.replace(bad, good)
    return text


def split_sentences(text: str):
    text = normalize(text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 25]


def keywords(text: str):
    """Important words of a text: no stop words, no duplicates, original order."""
    words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text.lower())
    seen, out = set(), []
    for w in words:
        if w not in STOP and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def top_keywords(text: str, n: int = 5):
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text.lower()) if w not in STOP]
    return [w for w, _ in Counter(words).most_common(n)]


# ---------------- Question generation (built from the document itself) ----------------

def good_subject(subject: str) -> bool:
    words = subject.split()
    if not words or len(words) > 4 or not CLEAN_SUBJECT.fullmatch(subject):
        return False
    for w in words:
        lw = w.lower()
        if lw in SKIP_SUBJECTS or lw in GENERIC or (lw.endswith("ly") and len(lw) > 4):
            return False
    return not words[0][0].isdigit()


def build_candidates(passage: str):
    """Turn sentences of the passage into questions using the templates."""
    questions = []
    for sentence in split_sentences(passage):
        for pattern, template in PATTERNS:
            m = pattern.match(sentence)
            if not m:
                continue
            subject = " ".join(m.group("s").split())          # remove extra spaces
            if not good_subject(subject):
                continue
            if subject.split()[0].lower() in {"a", "an", "the"}:
                subject = subject[0].lower() + subject[1:]
            verb = (m.groupdict().get("v") or "").lower()
            questions.append(template.format(s=subject, v=verb))
            break                                               # one question per sentence
    return questions


def next_question(session):
    """Return (question, passage). Every question comes from a sentence of the document."""
    chunks = chunk_document(session.document_text)
    if not chunks:
        raise ValueError("Document has too little text to ask questions.")

    used = [qa.ideal_answer for qa in session.qas.all()]
    asked = [qa.question for qa in session.qas.all()]
    fresh = [c for c in chunks if not any(c in u for u in used)] or chunks
    random.shuffle(fresh)

    # 1) template questions from passages not used yet
    for passage in fresh:
        options = [q for q in build_candidates(passage) if q not in asked]
        if options:
            return random.choice(options), passage

    # 2) fallback: ask about the most important term of a passage (always on topic)
    passage = fresh[0]
    for term in top_keywords(passage, 5):
        for q in (f"Explain {term} in your own words.", f"What does the document say about {term}?"):
            if q not in asked:
                return q, passage
    return f"Summarize this section in your own words: {passage[:60]}...", passage


# ---------------- Answer evaluation ----------------

def make_model_answer(question: str, passage: str) -> str:
    """Pick the 2 sentences of the passage that best match the question."""
    from sentence_transformers import util
    sentences = split_sentences(passage)
    if len(sentences) <= 2:
        return passage
    q_words = set(keywords(question))
    emb = get_embedder()
    sims = util.cos_sim(emb.encode(question, convert_to_tensor=True),
                        emb.encode(sentences, convert_to_tensor=True))[0].tolist()
    scores = []
    for sent, sim in zip(sentences, sims):
        overlap = len(q_words & set(keywords(sent))) / len(q_words) if q_words else 0
        scores.append(sim + 0.4 * overlap)           # meaning + exact word match
    top = sorted(sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:2])
    return " ".join(sentences[i] for i in top)


def score_answer(question: str, answer: str, passage: str) -> dict:
    """Score a user's answer from 0 to 10 against the document."""
    from sentence_transformers import util

    model_answer = make_model_answer(question, passage)
    q_words = set(keywords(question))
    keys = [k for k in keywords(model_answer) if k not in q_words][:8] or keywords(passage)[:8]

    # 1) meaning similarity with the correct answer and the passage
    emb = get_embedder()
    a, m, p = emb.encode([answer, model_answer, passage], convert_to_tensor=True)
    sim = max(float(util.cos_sim(a, m)), 0.9 * float(util.cos_sim(a, p)))
    sim_scaled = min(max((sim - 0.2) / 0.55, 0), 1)

    # 2) keyword coverage: how many key terms did the user mention?
    low = answer.lower()
    hit = [k for k in keys if k[:5] in low]
    missing = [k for k in keys if k[:5] not in low]
    coverage = len(hit) / len(keys) if keys else sim_scaled

    # 3) final score and verdict
    score = round(10 * (0.65 * sim_scaled + 0.35 * coverage), 1)
    verdict = "correct" if score >= 7 else "partial" if score >= 4 else "incorrect"

    return {"score": score, "verdict": verdict, "model_answer": model_answer,
            "missing": missing, "similarity": round(sim, 2), "coverage": round(coverage, 2)}


# ---------------- Feedback ----------------

def evaluate(question: str, answer: str, passage: str) -> dict:
    """Score the answer and build feedback. The views call this."""
    answer = answer.strip()

    # Case 1: the user does not know -> teach, don't score
    if answer.lower() in DONT_KNOW:
        model_answer = make_model_answer(question, passage)
        q_words = set(keywords(question))
        return {
            "score": 0,
            "verdict": "unknown",
            "feedback": "No problem. Not knowing is how you learn. Read the correct answer below, "
                        "then try to explain it in your own words.",
            "ideal_answer": f"{model_answer}\n\nFrom your document: {passage}",
            "tips": ["Read the correct answer once, close it, then say it aloud in 2-3 sentences.",
                     "In a real interview, say what you DO know first, then reason toward the answer."],
            "study_topics": [k for k in keywords(model_answer) if k not in q_words][:5],
        }

    # Case 2: the user answered -> score, then explain
    r = score_answer(question, answer, passage)
    missing = r["missing"]
    ideal = f"{r['model_answer']}\n\nFrom your document: {passage}"

    if r["verdict"] == "correct":
        feedback = "Great answer. It matches the document well."
        if missing:
            feedback += " To make it perfect, also mention: " + ", ".join(missing[:3]) + "."
        tips = ["Add a short real-life example to make your answer stand out in an interview."]
    elif r["verdict"] == "partial":
        feedback = "Good start, but some important points are missing."
        if missing:
            feedback += " Try to include: " + ", ".join(missing[:4]) + "."
        tips = ["Structure your answer: definition, how it works, then an example.",
                "Compare your answer with the correct answer below and add what is missing."]
    else:
        feedback = ("This doesn't match the document yet. Compare it with the correct answer "
                    "below and find the key idea you missed.")
        tips = ["Read the correct answer below, then explain it in your own words.",
                "Come back to this question after revising."]

    return {"score": r["score"], "verdict": r["verdict"], "feedback": feedback,
            "ideal_answer": ideal, "tips": tips, "study_topics": missing[:5]}


# ---------------- Final report ----------------

def summarize(session) -> dict:
    """Build the final report from all answered questions."""
    done = list(session.qas.filter(answered=True))
    avg = sum(q.score or 0 for q in done) / len(done)

    if avg >= 7:
        readiness = "Strong. You know this material well."
    elif avg >= 4:
        readiness = "Getting there. Revise the weak areas below."
    else:
        readiness = "Needs more study. Start from the basics of this document."

    def short(q):
        return q.question if len(q.question) < 90 else q.question[:87] + "..."

    weak = [q for q in done if (q.score or 0) < 7]
    topics = []
    for q in weak:
        topics += q.study_topics

    plan = [f"Re-read the section behind: {short(q)}" for q in weak]
    if topics:
        plan.append("Focus on these terms: " + ", ".join(dict.fromkeys(topics))[:200])
    plan.append("Retake this interview after revising and try to beat your score.")

    return {
        "readiness": readiness,
        "strengths": [short(q) for q in done if (q.score or 0) >= 7] or ["Keep practicing. Strengths will show up soon."],
        "weaknesses": [short(q) for q in done if (q.score or 0) < 4] or ["No major weak areas."],
        "study_plan": plan,
    }