-include ../tools.mk

export RUST_BACKTRACE=1
export RUST_LOG=rustc_save_analysis=debug
export RUST_SAVE_ANALYSIS_CONFIG={"output_file": null,"full_docs": true,"pub_only": false,"reachable_only": false,"distro_crate": false,"signatures": true,"borrow_data": true}

all: check
krate2: krate2.rs
	$(RUSTC) $<
code: foo.rs krate2
	$(RUSTC) foo.rs -Zsave-analysis 2>/dev/null
check: code
	mkdir -p $(TMPDIR)/sigs
	cat $(TMPDIR)/save-analysis/test.json | "$(PYTHON)" validate_json.py $(TMPDIR)/sigs
	@echo 'Checking that all signatures parse as valid rust code...'

	@for file in $$(ls $(TMPDIR)/sigs); do \
		echo Parsing signature $(TMPDIR)/sigs/$$file...; \
		$(RUSTC) -Zunstable-options --pretty=normal $(TMPDIR)/sigs/$$file ; \
	done

	exit 1