import json
from django.shortcuts import redirect, render
from django.utils.functional import Promise
from .apps import FastbertConfig
from django.http import JsonResponse
from django.http import HttpResponse
from PIL import Image
import torch 
import os
from django.core.files.storage import default_storage
from django.conf import settings
import tempfile
from django.views.decorators.csrf import csrf_exempt



def image_converstion(image):
    #  ""
    # args : image
    # use : convertin binary form
    # reutrn : binary object
    # """
            tupel_val = tempfile.mkstemp() 
            f = os.fdopen(tupel_val[0], 'wb')
            f.write(image.read()) 
            f.close()
            print(tupel_val[1])
            filepath = tupel_val[1]
            image = Image.open(filepath)
            image = image.convert('RGB')
            return image


def prediction_function(image):
    #  ""
    # args : binary image
    # use : predicting the image
    # reutrn : prediction array
    # """
            img_preprocessed = FastbertConfig.preprocess(image)
            batch_tensor = torch.unsqueeze(img_preprocessed, 0)
            FastbertConfig.resnet18.eval()
            prediction = FastbertConfig.resnet18(batch_tensor)
            return prediction

def helper(prediction):
      #  ""
    # args : prediction of image
    # use :  converting binary prediction in readable format
    # reutrn : returning the prediction of image
    # """

     with open(str(settings.BASE_DIR)+'/imagenet_classes.txt') as f:

        labels = [line.strip() for line in f.readlines()]
            # _, index = torch.max(prediction, 1)
        percentage = torch.nn.functional.softmax(prediction, dim=1)[0] * 100
        _, indices = torch.sort(prediction, descending=True)

     answer = []
     for idx in indices[0][:1] :
         answer.append(labels[idx])
         answer.append(str(round(percentage[idx].item(),3)))
     return answer
    
    
    

@csrf_exempt
def call_model(request) :

    #  ""
    # args : http post request
    # use : to predict the image
    # reutrn : closest prediction of uploaded image
    # """
    
    if request.method == 'POST' and request.FILES['myfile']:

            image = request.FILES.get('myfile') 
            
            image = image_converstion(image)
            
            prediction = prediction_function(image)
            

            answer = helper(prediction)
        
            return render(request, "result.html", {'val1':answer[0] , 'val2':answer[1] } )

    else :
       return HttpResponse("BAD Request")

