from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('performance/', views.performance, name='performance'),
    path('field/', views.field, name='field'),
    path('gpa/', views.gpa, name='gpa'),
]