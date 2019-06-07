-include ../tools.mk

export RUST_BACKTRACE=1
export RUSTC_LOG=rustc_save_analysis=debug
export RUST_SAVE_ANALYSIS_CONFIG={"output_file": null,"full_docs": true,"pub_only": false,"reachable_only": false,"distro_crate": false,"signatures": true,"borrow_data": true}

all: check
krate2: krate2.rs
	$(RUSTC) $<
code: foo.rs krate2
	$(RUSTC) foo.rs -Zsave-analysis
check: code
	rm -rf $(TMPDIR)/sigs
	mkdir -p $(TMPDIR)/sigs
	cat $(TMPDIR)/save-analysis/test.json | "$(PYTHON)" validate_json.py $(TMPDIR)/sigs
	@echo 'Checking that all signatures parse as valid rust code...'

	@for file in $$(ls $(TMPDIR)/sigs); do \
		$(RUSTC) -Zunstable-options --pretty=normal $(TMPDIR)/sigs/$$file > /dev/null 2>$(TMPDIR)/sigs/$$file.out & \
	done; \
	wait; \
	for file in $$(ls $(TMPDIR)/sigs | grep .out | sort); do \
		cat $(TMPDIR)/sigs/$$file; \
	done