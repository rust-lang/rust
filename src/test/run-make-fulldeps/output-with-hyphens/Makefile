include ../tools.mk

all:
	$(RUSTC) foo-bar.rs --crate-type bin
	[ -f $(TMPDIR)/$(call BIN,foo-bar) ]
	$(RUSTC) foo-bar.rs --crate-type lib
	[ -f $(TMPDIR)/libfoo_bar.rlib ]
