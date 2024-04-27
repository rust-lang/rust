# ignore-cross-compile
include ../tools.mk

all:
	$(RUSTC) foo.rs
	$(RUSTC) -Z ls=root $(TMPDIR)/foo
	touch $(TMPDIR)/bar
	$(RUSTC) -Z ls=root $(TMPDIR)/bar
