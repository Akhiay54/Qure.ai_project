from django.apps import AppConfig
from torchvision import models
from torchvision import transforms

class FastbertConfig(AppConfig):
    #default_auto_field = 'django.db.models.BigAutoField'
    name = 'fastbert'
    
    ## model calling-------------------------
    resnet18 = models.resnet18(pretrained=True)
    preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

