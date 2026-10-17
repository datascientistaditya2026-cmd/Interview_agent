from django.db import models


class Session(models.Model):
    role = models.CharField(max_length=120)             # e.g. "Python Django Developer"
    topic = models.CharField(max_length=200, blank=True)
    document_text = models.TextField(blank=True)        # full text of the uploaded document
    total_questions = models.PositiveIntegerField(default=5)
    summary = models.JSONField(null=True, blank=True)   # final report, filled at the end
    created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.role} - {self.topic or 'general'}"


class QA(models.Model):
    session = models.ForeignKey(Session, related_name="qas", on_delete=models.CASCADE)
    question = models.TextField()
    answer = models.TextField(blank=True)
    answered = models.BooleanField(default=False)
    score = models.FloatField(null=True, blank=True)       # 0 to 10
    verdict = models.CharField(max_length=20, blank=True)  # correct / partial / incorrect / unknown
    feedback = models.TextField(blank=True)
    ideal_answer = models.TextField(blank=True)            # also holds the source passage before evaluation
    tips = models.JSONField(default=list)                  # how to improve
    study_topics = models.JSONField(default=list)          # what to study next

    class Meta:
        ordering = ["id"]

    def __str__(self):
        return self.question[:60]