from django.contrib import admin
from .models import Company, JobPosition, Application

admin.site.register(Company)
admin.site.register(JobPosition)
admin.site.register(Application)
