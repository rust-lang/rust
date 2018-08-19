-include ../tools.mk

all: extern_absolute_paths.rs extern_in_paths.rs krate2
	$(RUSTC) extern_absolute_paths.rs -Zsave-analysis --edition=2018
	cat $(TMPDIR)/save-analysis/extern_absolute_paths.json | "$(PYTHON)" validate_json.py
	$(RUSTC) extern_in_paths.rs -Zsave-analysis --edition=2018
	cat $(TMPDIR)/save-analysis/extern_in_paths.json | "$(PYTHON)" validate_json.py

krate2: krate2.rs
	$(RUSTC) $<
