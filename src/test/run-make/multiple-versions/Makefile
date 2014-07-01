-include ../tools.mk

all:
	$(RUSTC) foo.rs -C metadata=a -C extra-filename=-1 --crate-type=rlib
	$(RUSTC) foo.rs -C metadata=b -C extra-filename=-2 --crate-type=rlib
	$(RUSTC) bar.rs \
		--extern foo1=$(TMPDIR)/libfoo-1.rlib \
		--extern foo2=$(TMPDIR)/libfoo-2.rlib \
		2>&1 | grep "using multiple versions of crate .foo."
