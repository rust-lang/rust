include ../../run-make-fulldeps/tools.mk

# Regression test for issue #10971
# Running two invocations in parallel would overwrite each other's temp files.

all:
	touch $(TMPDIR)/lib.rs

	$(RUSTC) --crate-type=lib --temps-dir=$(TMPDIR)/temp1 $(TMPDIR)/lib.rs & \
	$(RUSTC) --crate-type=cdylib --temps-dir=$(TMPDIR)/temp2 $(TMPDIR)/lib.rs
