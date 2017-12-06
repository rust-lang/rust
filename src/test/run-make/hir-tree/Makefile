-include ../tools.mk

# Test that hir-tree output doens't crash and includes
# the string constant we would expect to see.

all:
	$(RUSTC) -o $(TMPDIR)/input.hir -Z unstable-options \
		--unpretty=hir-tree input.rs
	$(CGREP) '"Hello, Rustaceans!\n"' < $(TMPDIR)/input.hir
