from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
import numpy as np
import os
from configs.settings import BASE_DIR
import matplotlib
import matplotlib
matplotlib.use('agg')
import tensorflow as tf
import cv2
from django.http import JsonResponse
from django.utils.decorators import method_decorator

@csrf_exempt
def verify_papaya_dynamic_view(request):
    '''Docstring here.'''

    
    context = {
    }
    return render(
        request,
        'home/index_verify_dynamic.html',
        context,
    )
