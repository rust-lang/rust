-include ../../run-make-fulldeps/tools.mk

# Regression test for issue #10971
# Running two invocations in parallel would overwrite each other's temp files.

all:
	touch $(TMPDIR)/lib.rs

	$(RUSTC) --crate-type=lib -Z temps-dir=$(TMPDIR)/temp1 $(TMPDIR)/lib.rs & \
	$(RUSTC) --crate-type=staticlib -Z temps-dir=$(TMPDIR)/temp2 $(TMPDIR)/lib.rs
