from django.shortcuts import render
from .prediction import performance_predict, field_predict, gpa_predict

def index(request):
	context = {}
	return render(request, 'spfp/index.html', context)

def performance(request):
	features = ['sex', 'pstatus', 'medu', 'fedu', 'peer', 'region', 'studytime', 'math',
		 'phy', 'che', 'bio', 'eng', 'civic', 'aptitude', 'gche', 'ceng', 'gphy1',
		  'amath1', 'python', 'introcivic', 'amath2', 'gphy2', 'cpp', 'bwrite', 'draw', 'logic']
	
	context = {'performance': None}
	if request.method == 'POST':
		sample = []
		for f in features:
			sample.append(int(request.POST.get(f, False)))
		result = performance_predict(sample)
		context['performance'] = result	
	
	return render(request, 'spfp/performance.html', context)

def field(request):
	context = {'field': None}
	features = ['sex', 'pstatus', 'medu', 'fedu', 'peer', 'region', 'studytime', 'math',
		 'phy', 'che', 'bio', 'eng', 'civic', 'aptitude', 'gche', 'ceng', 'gphy1',
		  'amath1', 'python', 'introcivic', 'amath2', 'gphy2', 'cpp', 'bwrite', 'draw', 'logic']
	
	if request.method == 'POST':
		sample = []
		for f in features:
			sample.append(int(request.POST.get(f, False)))
		result = field_predict(sample)
		context['field'] = result	
	
	return render(request, 'spfp/field.html', context)

def gpa(request):
	features = ['sex', 'pstatus', 'medu', 'fedu', 'peer', 'region', 'studytime', 'math',
		 'phy', 'che', 'bio', 'eng', 'civic', 'aptitude']
	
	context = {'gpa': None}
	if request.method == 'POST':
		sample = []
		for f in features:
			sample.append(int(request.POST.get(f, False)))
		result = gpa_predict(sample)
		context['gpa'] = result	
		
	return render(request, 'spfp/gpa.html', context)