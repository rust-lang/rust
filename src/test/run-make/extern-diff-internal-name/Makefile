-include ../tools.mk

all:
	$(RUSTC) lib.rs
	$(RUSTC) test.rs --extern foo=$(TMPDIR)/libbar.rlib 2>&1 | \
		{ ! grep "using multiple versions of crate"; }
