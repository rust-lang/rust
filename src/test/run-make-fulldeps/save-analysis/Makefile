-include ../tools.mk
all: check
krate2: krate2.rs
	$(RUSTC) $<
code: foo.rs krate2
	$(RUSTC) foo.rs -Zsave-analysis
check: code
	mkdir -p $(TMPDIR)/intermediate
	cat $(TMPDIR)/save-analysis/test.json | "$(PYTHON)" validate_json.py $(TMPDIR)/sigs
	$(foreach sig,$(wildcard $(TMPDIR)/sigs/*),rustc -Z unstable-options --pretty normal $(sig))