include ../tools.mk

all:
	$(RUSTC) libc.rs -Cmetadata=foo
	$(RUSTC) main.rs --extern libc=$(TMPDIR)/liblibc.rlib
