-include ../tools.mk

all:
	$(RUSTC) bar.rs --crate-type=rlib
	$(RUSTC) bar.rs --crate-type=rlib -C extra-filename=-a
	$(RUSTC) foo.rs --extern hello && exit 1 || exit 0
	$(RUSTC) foo.rs --extern bar=no-exist && exit 1 || exit 0
	$(RUSTC) foo.rs --extern bar=foo.rs && exit 1 || exit 0
	$(RUSTC) foo.rs \
		--extern bar=$(TMPDIR)/libbar.rlib \
		--extern bar=$(TMPDIR)/libbar-a.rlib \
		&& exit 1 || exit 0
	$(RUSTC) foo.rs \
		--extern bar=$(TMPDIR)/libbar.rlib \
		--extern bar=$(TMPDIR)/libbar.rlib
	$(RUSTC) foo.rs --extern bar=$(TMPDIR)/libbar.rlib
