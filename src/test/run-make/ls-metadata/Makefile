-include ../tools.mk

all:
	$(RUSTC) foo.rs
	$(RUSTC) --ls $(TMPDIR)/foo
	touch $(TMPDIR)/bar
	$(RUSTC) --ls $(TMPDIR)/bar
