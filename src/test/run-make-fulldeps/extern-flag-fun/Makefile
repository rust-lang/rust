include ../tools.mk

all:
	$(RUSTC) bar.rs --crate-type=rlib
	$(RUSTC) bar.rs --crate-type=rlib -C extra-filename=-a
	$(RUSTC) bar-alt.rs --crate-type=rlib
	$(RUSTC) foo.rs --extern bar=no-exist && exit 1 || exit 0
	$(RUSTC) foo.rs --extern bar=foo.rs && exit 1 || exit 0
	$(RUSTC) foo.rs \
		--extern bar=$(TMPDIR)/libbar.rlib \
		--extern bar=$(TMPDIR)/libbar-alt.rlib \
		&& exit 1 || exit 0
	$(RUSTC) foo.rs \
		--extern bar=$(TMPDIR)/libbar.rlib \
		--extern bar=$(TMPDIR)/libbar-a.rlib
	$(RUSTC) foo.rs --extern bar=$(TMPDIR)/libbar.rlib
	# Try to be sneaky and load a private crate from with a non-private name.
	$(RUSTC) rustc.rs -Zforce-unstable-if-unmarked --crate-type=rlib
	$(RUSTC) gated_unstable.rs --extern alloc=$(TMPDIR)/librustc.rlib 2>&1 | $(CGREP) 'rustc_private'
