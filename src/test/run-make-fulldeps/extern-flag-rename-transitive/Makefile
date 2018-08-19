-include ../tools.mk

all:
	$(RUSTC) foo.rs
	$(RUSTC) bar.rs
	$(RUSTC) baz.rs --extern a=$(TMPDIR)/libfoo.rlib

