include ../tools.mk

# ignore-windows
# `ln` is actually `cp` on msys.

all:
	$(RUSTC) foo.rs
	mkdir -p $(TMPDIR)/other
	ln -nsf $(TMPDIR)/libfoo.rlib $(TMPDIR)/other
	$(RUSTC) bar.rs -L $(TMPDIR)
	$(RUSTC) baz.rs --extern foo=$(TMPDIR)/other/libfoo.rlib  -L $(TMPDIR)
