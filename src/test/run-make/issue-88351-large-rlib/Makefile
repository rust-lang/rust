-include ../../run-make-fulldeps/tools.mk

all:
	$(RUSTC) main.rs
	$(TMPDIR)/main $(TMPDIR)
	$(RUSTC) $(TMPDIR)/foo.rs --crate-type=rlib -l static=foo -L$(TMPDIR)
	RUSTC_LOG=rustc_metadata=debug $(RUSTC) bar.rs --extern foo=$(TMPDIR)/libfoo.rlib --edition=2018
