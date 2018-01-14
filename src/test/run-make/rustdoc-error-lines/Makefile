-include ../tools.mk

# Test that hir-tree output doens't crash and includes
# the string constant we would expect to see.

all:
	$(RUSTDOC) --test input.rs > $(TMPDIR)/output || true
	$(CGREP) 'input.rs:17:15' < $(TMPDIR)/output
