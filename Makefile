venv:
	python3 -m venv ~/.pca
	#source ~/.pca/bin/activate

install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

lint:
	pylint --disable=R,C mnist