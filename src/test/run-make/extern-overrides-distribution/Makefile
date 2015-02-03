-include ../tools.mk

all:
	$(RUSTC) libc.rs
	$(RUSTC) main.rs --extern libc=$(TMPDIR)/liblibc.rlib
