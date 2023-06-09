# ignore-cross-compile
include ../tools.mk

# Test expected libtest's junit output

OUTPUT_FILE_DEFAULT := $(TMPDIR)/libtest-junit-output-default.xml
OUTPUT_FILE_STDOUT_SUCCESS := $(TMPDIR)/libtest-junit-output-stdout-success.xml

all: f.rs validate_junit.py output-default.xml output-stdout-success.xml
	$(RUSTC) --test f.rs
	RUST_BACKTRACE=0 $(call RUN,f) -Z unstable-options --test-threads=1 --format=junit > $(OUTPUT_FILE_DEFAULT) || true
	RUST_BACKTRACE=0 $(call RUN,f) -Z unstable-options --test-threads=1 --format=junit --show-output > $(OUTPUT_FILE_STDOUT_SUCCESS) || true

	cat $(OUTPUT_FILE_DEFAULT) | "$(PYTHON)" validate_junit.py
	cat $(OUTPUT_FILE_STDOUT_SUCCESS) | "$(PYTHON)" validate_junit.py

	# Normalize the actual output and compare to expected output file
	cat $(OUTPUT_FILE_DEFAULT) | sed 's/time="[0-9.]*"/time="$$TIME"/g' | diff output-default.xml -
	cat $(OUTPUT_FILE_STDOUT_SUCCESS) | sed 's/time="[0-9.]*"/time="$$TIME"/g' | diff output-stdout-success.xml -
