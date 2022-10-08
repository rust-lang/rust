# Staticlib format is no longer checked during bundling

include ../tools.mk

all:
	touch $(TMPDIR)/libfoo.a
	$(RUSTC) lib.rs --crate-type=rlib -lstatic=foo
