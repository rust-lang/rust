# This test checks that extern crate declarations in Cargo without a corresponding declaration in the manifest of a dependency are NOT allowed.
# See https://github.com/rust-lang/rust/pull/21113

include ../tools.mk

all:
	mkdir -p $(TMPDIR)/a $(TMPDIR)/b
	$(RUSTC) a.rs && mv $(TMPDIR)/liba.rlib $(TMPDIR)/a
	$(RUSTC) b.rs -L $(TMPDIR)/a && mv $(TMPDIR)/libb.rlib $(TMPDIR)/b
	$(RUSTC) c.rs -L crate=$(TMPDIR)/b -L dependency=$(TMPDIR)/a \
		&& exit 1 || exit 0
