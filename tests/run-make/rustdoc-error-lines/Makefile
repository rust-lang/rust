include ../tools.mk

# Test that hir-tree output doesn't crash and includes
# the string constant we would expect to see.

all:
	$(RUSTDOC) --test input.rs > $(TMPDIR)/output || true
	$(CGREP) 'input.rs - foo (line 5)' < $(TMPDIR)/output
	$(CGREP) 'input.rs:7:15' < $(TMPDIR)/output
	$(CGREP) 'input.rs - bar (line 15)' < $(TMPDIR)/output
	$(CGREP) 'input.rs:17:15' < $(TMPDIR)/output
	$(CGREP) 'input.rs - bar (line 24)' < $(TMPDIR)/output
	$(CGREP) 'input.rs:26:15' < $(TMPDIR)/output
