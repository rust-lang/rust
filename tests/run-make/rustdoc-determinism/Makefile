include ../tools.mk

# Assert that the search index is generated deterministically, regardless of the
# order that crates are documented in.

# ignore-windows
# Uses `diff`.

all:
	$(RUSTDOC) foo.rs -o $(TMPDIR)/foo_first
	$(RUSTDOC) bar.rs -o $(TMPDIR)/foo_first

	$(RUSTDOC) bar.rs -o $(TMPDIR)/bar_first
	$(RUSTDOC) foo.rs -o $(TMPDIR)/bar_first

	diff $(TMPDIR)/foo_first/search-index.js $(TMPDIR)/bar_first/search-index.js
