import requests
from django.shortcuts import redirect, render 
from django.core.files.storage import FileSystemStorage


def home(request):
    return render(request,'home.html')
