from django.contrib import admin
from django.urls import path
from project import views

urlpatterns = [
    path('', views.user , name = 'user'),
    path('viewdata', views.viewdata , name = 'viewdata'),
]