# Generated by Django 4.2.17 on 2024-12-27 17:05

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import django_extensions.db.fields


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('jobs', '0002_remove_cv_matched_skills_count'),
    ]

    operations = [
        migrations.CreateModel(
            name='Application',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created', django_extensions.db.fields.CreationDateTimeField(auto_now_add=True, verbose_name='created')),
                ('modified', django_extensions.db.fields.ModificationDateTimeField(auto_now=True, verbose_name='modified')),
                ('applicant_name', models.CharField(max_length=255)),
                ('applicant_email', models.EmailField(max_length=254)),
                ('cv', models.FileField(upload_to='cvs/')),
                ('cover_letter', models.TextField(blank=True, null=True)),
                ('status', models.CharField(choices=[('P', 'Pending'), ('R', 'Reviewed'), ('A', 'Accepted'), ('D', 'Declined')], default='P', max_length=50)),
                ('extracted_text', models.TextField(blank=True, null=True)),
            ],
            options={
                'get_latest_by': 'modified',
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='Company',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created', django_extensions.db.fields.CreationDateTimeField(auto_now_add=True, verbose_name='created')),
                ('modified', django_extensions.db.fields.ModificationDateTimeField(auto_now=True, verbose_name='modified')),
                ('name', models.CharField(max_length=255)),
                ('location', models.CharField(max_length=255)),
                ('description', models.TextField(blank=True, null=True)),
                ('website', models.URLField(blank=True, null=True)),
                ('established_date', models.DateField(blank=True, null=True)),
                ('admin', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='companies', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'get_latest_by': 'modified',
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='JobPosition',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created', django_extensions.db.fields.CreationDateTimeField(auto_now_add=True, verbose_name='created')),
                ('modified', django_extensions.db.fields.ModificationDateTimeField(auto_now=True, verbose_name='modified')),
                ('title', models.CharField(max_length=255)),
                ('description', models.TextField()),
                ('required_skills', models.TextField()),
                ('application_deadline', models.DateField(blank=True, null=True)),
                ('salary_range', models.CharField(blank=True, max_length=100, null=True)),
                ('job_type', models.CharField(choices=[('FT', 'Full-Time'), ('PT', 'Part-Time'), ('CT', 'Contract')], max_length=50)),
                ('company', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='job_positions', to='jobs.company')),
            ],
            options={
                'get_latest_by': 'modified',
                'abstract': False,
            },
        ),
        migrations.DeleteModel(
            name='CV',
        ),
        migrations.DeleteModel(
            name='JobPost',
        ),
        migrations.AddField(
            model_name='application',
            name='job_position',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='applications', to='jobs.jobposition'),
        ),
    ]
