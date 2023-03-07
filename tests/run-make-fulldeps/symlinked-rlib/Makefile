include ../tools.mk

# ignore-windows
# `ln` is actually `cp` on msys.

all:
	$(RUSTC) foo.rs --crate-type=rlib -o $(TMPDIR)/foo.xxx
	ln -nsf $(TMPDIR)/foo.xxx $(TMPDIR)/libfoo.rlib
	$(RUSTC) bar.rs -L $(TMPDIR)
