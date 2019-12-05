-include ../tools.mk

# Test --extern noprelude

all:
	$(RUSTC) dep.rs --crate-name=dep --crate-type=rlib
	$(RUSTC) foo.rs --edition=2018 -Zunstable-options \
		--extern noprelude:dep=$(TMPDIR)/libdep.rlib 2>&1 | \
		$(CGREP) -e 'failed to resolve.*`dep`'
	$(RUSTC) foo.rs --edition=2018 -Zunstable-options \
		--extern dep=$(TMPDIR)/libdep.rlib
