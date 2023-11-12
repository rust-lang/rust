
include ../tools.mk

all: inconsistent-mapping last-crate-only

last-crate-only:
	$(RUSTC) is_true.rs --crate-type lib
	$(RUSTC) main.rs --remap-path-prefix=$(CURDIR)=/TEST_DIR 2>$(TMPDIR)/last-crate.txt --extern is_true=$(TMPDIR)/libis_true.rlib || true
	$(RUSTC_TEST_OP) $(TMPDIR)/last-crate.txt expected-last-crate.txt
	$(CGREP) "/TEST_DIR" < $(TMPDIR)/last-crate.txt

inconsistent-mapping:
	$(RUSTC) is_true.rs --remap-path-prefix=$(CURDIR)=/TEST_DIR1 --crate-type lib
	$(RUSTC) main.rs --remap-path-prefix=$(CURDIR)=/TEST_DIR2 2>$(TMPDIR)/inconsistent.txt --extern is_true=$(TMPDIR)/libis_true.rlib || true
	$(RUSTC_TEST_OP) $(TMPDIR)/inconsistent.txt expected-inconsistent.txt
	$(CGREP) "/TEST_DIR1" < $(TMPDIR)/inconsistent.txt
