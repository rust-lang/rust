-include ../tools.mk

# Test expected libtest's JSON output

OUTPUT_FILE := $(TMPDIR)/libtest-json-output.json

all:
	$(RUSTC) --test f.rs
	RUST_BACKTRACE=0 $(call RUN,f) -Z unstable-options --test-threads=1 --format=json > $(OUTPUT_FILE) || true

	cat $(OUTPUT_FILE) | "$(PYTHON)" validate_json.py

	# Compare to output file
	diff output.json $(OUTPUT_FILE)
