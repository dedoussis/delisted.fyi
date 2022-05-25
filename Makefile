clean:
	find . -name "*.py[cod]" -delete

format:
	black .

typecheck:
	mypy --package delisted --strict

install: clean
	poetry install

.PHONY: clean format install typecheck
