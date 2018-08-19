-include ../tools.mk

all:
	$(RUSTC) foo-bar.rs
	[ -f $(TMPDIR)/$(call BIN,foo-bar) ]
	[ -f $(TMPDIR)/libfoo_bar.rlib ]
