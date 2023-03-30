# ignore-cross-compile
include ../tools.mk

all:
	$(RUSTC) lib.rs
	$(RUSTC) test.rs --extern foo=$(TMPDIR)/libbar.rlib
