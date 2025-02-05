# only-x86_64-unknown-linux-gnu

include ../tools.mk

# assert that import list for simple executable not changed silently
# requires binutils

all:
	$(RUSTC) empty.rs
	readelf --dyn-syms -W $(TMPDIR)/empty > "$(TMPDIR)/actual.empty.txt"
	diff -u expected.empty.txt "$(TMPDIR)/actual.empty.txt"
