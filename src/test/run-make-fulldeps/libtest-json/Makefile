include ../tools.mk

# Test expected libtest's JSON output

OUTPUT_FILE_DEFAULT := $(TMPDIR)/libtest-json-output-default.json
OUTPUT_FILE_STDOUT_SUCCESS := $(TMPDIR)/libtest-json-output-stdout-success.json

all: f.rs validate_json.py output-default.json output-stdout-success.json
	$(RUSTC) --test f.rs
	RUST_BACKTRACE=0 $(call RUN,f) -Z unstable-options --test-threads=1 --format=json > $(OUTPUT_FILE_DEFAULT) || true
	RUST_BACKTRACE=0 $(call RUN,f) -Z unstable-options --test-threads=1 --format=json --show-output > $(OUTPUT_FILE_STDOUT_SUCCESS) || true

	cat $(OUTPUT_FILE_DEFAULT) | "$(PYTHON)" validate_json.py
	cat $(OUTPUT_FILE_STDOUT_SUCCESS) | "$(PYTHON)" validate_json.py

	# Normalize the actual output and compare to expected output file
	cat $(OUTPUT_FILE_DEFAULT) | sed 's/"exec_time": [0-9.]*/"exec_time": $$TIME/' | diff output-default.json -
	cat $(OUTPUT_FILE_STDOUT_SUCCESS) | sed 's/"exec_time": [0-9.]*/"exec_time": $$TIME/' | diff output-stdout-success.json -
