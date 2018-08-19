-include ../tools.mk

all:
	$(RUSTC) -C extra-filename=-hash foo.rs
	$(RUSTC) bar.rs
	mv $(TMPDIR)/libfoo-hash.rlib $(TMPDIR)/libfoo-another-hash.rlib
	$(RUSTC) baz.rs
