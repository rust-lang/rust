-include ../tools.mk

all:
	$(RUSTC) -Z no-trans -Z unstable-options \
		--complete-at input.rs:583 input.rs > $(TMPDIR)/output
	cmp output $(TMPDIR)/output
