clean:
	find . -name "*.py[cod]" -delete

format:
	black .

install: clean
	poetry install

.PHONY: clean format install
