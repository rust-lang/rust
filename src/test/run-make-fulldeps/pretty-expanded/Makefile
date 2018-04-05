-include ../tools.mk

all:
	$(RUSTC) -o $(TMPDIR)/input.expanded.rs -Z unstable-options \
		--pretty=expanded input.rs
