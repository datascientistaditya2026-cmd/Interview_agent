from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("session/<int:pk>/voice/", views.voice_view, name="voice"),
    path("session/<int:pk>/api/next/", views.api_next, name="api_next"),
    path("session/<int:pk>/api/answer/", views.api_answer, name="api_answer"),
    path("session/<int:pk>/", views.session_view, name="session"),
    path("session/<int:pk>/summary/", views.summary, name="summary"),
]