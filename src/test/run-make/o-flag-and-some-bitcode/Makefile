-include ../tools.mk

all:
	$(RUSTC) foo.rs -o $(TMPDIR)/libfoo-somehash-0.0.rlib
	$(RUSTC) main.rs -Zlto
