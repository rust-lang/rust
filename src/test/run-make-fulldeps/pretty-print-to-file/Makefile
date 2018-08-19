-include ../tools.mk

all:
	$(RUSTC) -o $(TMPDIR)/input.out --pretty=normal -Z unstable-options input.rs
	diff -u $(TMPDIR)/input.out input.pp
