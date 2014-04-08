-include ../tools.mk

all:
	$(RUSTC) -o $(TMPDIR)/input.out --pretty=normal input.rs
	diff -u $(TMPDIR)/input.out input.pp
