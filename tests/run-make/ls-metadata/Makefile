# ignore-cross-compile
include ../tools.mk

all:
	$(RUSTC) foo.rs
	$(RUSTC) -Z ls $(TMPDIR)/foo
	touch $(TMPDIR)/bar
	$(RUSTC) -Z ls $(TMPDIR)/bar
