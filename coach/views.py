from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.views.decorators.http import require_POST

from . import agent
from .documents import read_document
from .models import Session, QA


def home(request):
    """Start page: role + document upload. Creates a Session."""
    error = ""
    if request.method == "POST":
        role = request.POST.get("role", "").strip()
        topic = request.POST.get("topic", "").strip()
        doc = request.FILES.get("document")

        if not role or not doc:
            error = "Enter a role and upload a document (.pdf, .txt, .md)."
        else:
            text = read_document(doc)
            if len(text.strip()) < 200:
                error = "Could not read enough text from this file. Try a text-based PDF or a .txt file."
            else:
                s = Session.objects.create(
                    role=role, topic=topic, document_text=text,
                    total_questions=int(request.POST.get("total", 5)),
                )
                return redirect("voice", pk=s.pk)        # go to the voice interview

    return render(request, "coach/home.html", {"error": error})


# ---------------- shared helper ----------------

def save_evaluation(qa, answer, r):
    qa.answer = answer
    qa.answered = True
    qa.score = r["score"]
    qa.verdict = r["verdict"]
    qa.feedback = r["feedback"]
    qa.ideal_answer = r["ideal_answer"]       # now: correct answer + passage
    qa.tips = r["tips"]
    qa.study_topics = r["study_topics"]
    qa.save()


# ---------------- Voice interview ----------------

def voice_view(request, pk):
    s = get_object_or_404(Session, pk=pk)
    return render(request, "coach/voice.html", {"s": s})


def api_next(request, pk):
    """Return the next question as JSON (or the final result when all questions are done)."""
    s = get_object_or_404(Session, pk=pk)
    answered = s.qas.filter(answered=True).count()

    if answered >= s.total_questions:
        if not s.summary:
            s.summary = agent.summarize(s)
            s.save()
        done = s.qas.filter(answered=True)
        avg = round(sum(q.score or 0 for q in done) / done.count(), 1)
        return JsonResponse({"done": True, "avg": avg, "readiness": s.summary["readiness"],
                             "summary_url": reverse("summary", args=[s.pk])})

    qa = s.qas.filter(answered=False).first()
    if not qa:
        question, passage = agent.next_question(s)
        qa = QA.objects.create(session=s, question=question, ideal_answer=passage)

    return JsonResponse({"done": False, "qa_id": qa.pk, "question": qa.question,
                         "number": answered + 1, "total": s.total_questions})


@require_POST
def api_answer(request, pk):
    """Evaluate a spoken answer and return the feedback as JSON."""
    s = get_object_or_404(Session, pk=pk)
    qa = get_object_or_404(QA, pk=request.POST.get("qa_id"), session=s, answered=False)
    answer = request.POST.get("answer", "")

    r = agent.evaluate(qa.question, answer, qa.ideal_answer)   # ideal_answer still holds the passage here
    save_evaluation(qa, answer, r)

    return JsonResponse({
        "score": r["score"],
        "verdict": r["verdict"],
        "feedback": r["feedback"],
        "model_answer": r["ideal_answer"].split("\n\nFrom your document:")[0],
        "study_topics": r["study_topics"],
    })


# ---------------- Original chat version (still available) ----------------

def session_view(request, pk):
    s = get_object_or_404(Session, pk=pk)

    if request.method == "POST":
        qa = get_object_or_404(QA, pk=request.POST["qa_id"], session=s, answered=False)
        answer = "" if "dont_know" in request.POST else request.POST.get("answer", "")
        r = agent.evaluate(qa.question, answer, qa.ideal_answer)
        save_evaluation(qa, answer, r)
        return redirect(f"/session/{s.pk}/?fb={qa.pk}")

    fb_id = request.GET.get("fb")
    if fb_id:
        qa = get_object_or_404(QA, pk=fb_id, session=s)
        finished = s.qas.filter(answered=True).count() >= s.total_questions
        return render(request, "coach/feedback.html", {"s": s, "qa": qa, "finished": finished})

    answered_count = s.qas.filter(answered=True).count()
    if answered_count >= s.total_questions:
        return redirect("summary", pk=s.pk)

    qa = s.qas.filter(answered=False).first()
    if not qa:
        question, passage = agent.next_question(s)
        qa = QA.objects.create(session=s, question=question, ideal_answer=passage)
    return render(request, "coach/question.html", {"s": s, "qa": qa, "number": answered_count + 1})


def summary(request, pk):
    s = get_object_or_404(Session, pk=pk)
    answered = s.qas.filter(answered=True)
    if not answered.exists():
        return redirect("session", pk=s.pk)

    if not s.summary:
        s.summary = agent.summarize(s)
        s.save()

    avg = round(sum(q.score or 0 for q in answered) / answered.count(), 1)
    return render(request, "coach/summary.html", {"s": s, "qas": answered, "avg": avg})