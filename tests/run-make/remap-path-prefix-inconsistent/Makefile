
include ../tools.mk

all: inconsistent-mapping last-crate-only remap-scope

last-crate-only:
	$(RUSTC) is_true.rs --crate-type lib
	$(RUSTC) main.rs --remap-path-prefix=$(CURDIR)=/TEST_DIR 2>$(TMPDIR)/last-crate.txt --extern is_true=$(TMPDIR)/libis_true.rlib || true
	$(RUSTC_TEST_OP) $(TMPDIR)/last-crate.txt expected-last-crate.txt
	$(CGREP) "/TEST_DIR" < $(TMPDIR)/last-crate.txt

# exactly like `last-crate-only`, but also uses `-Z remap-path-scope=diagnostics`
remap-scope:
	$(RUSTC) is_true.rs --crate-type lib
	$(RUSTC) main.rs -Z remap-path-scope=diagnostics --remap-path-prefix=$(CURDIR)=/TEST_DIR 2>$(TMPDIR)/last-crate.txt --extern is_true=$(TMPDIR)/libis_true.rlib || true
	$(RUSTC_TEST_OP) $(TMPDIR)/last-crate.txt expected-last-crate.txt
	$(CGREP) "/TEST_DIR" < $(TMPDIR)/last-crate.txt
	# Now again, but excluding diagnostics
	$(RUSTC) main.rs -Z remap-path-scope=object --remap-path-prefix=$(CURDIR)=/TEST_DIR --extern is_true=$(TMPDIR)/libis_true.rlib 2>&1 | $(CGREP) tests/run-make/

inconsistent-mapping:
	$(RUSTC) is_true.rs --remap-path-prefix=$(CURDIR)=/TEST_DIR1 --crate-type lib
	$(RUSTC) main.rs --remap-path-prefix=$(CURDIR)=/TEST_DIR2 2>$(TMPDIR)/inconsistent.txt --extern is_true=$(TMPDIR)/libis_true.rlib || true
	$(RUSTC_TEST_OP) $(TMPDIR)/inconsistent.txt expected-inconsistent.txt
	$(CGREP) "/TEST_DIR1" < $(TMPDIR)/inconsistent.txt
