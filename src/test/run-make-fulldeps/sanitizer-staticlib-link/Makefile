# needs-sanitizer-support
# only-x86_64
-include ../tools.mk

# This test builds a staticlib, then an executable that links to it.
# The staticlib and executable are compiled with address sanitizer, and we
# assert that a fault in the staticlib is correctly detected.

all:
	$(RUSTC) -g -Z sanitizer=address --crate-type staticlib --target $(TARGET) library.rs
	$(RUSTC) -g -Z sanitizer=address --crate-type bin --target $(TARGET) $(EXTRA_RUSTFLAG) -llibrary program.rs
	LD_LIBRARY_PATH=$(TMPDIR) $(TMPDIR)/program 2>&1 | $(CGREP) stack-buffer-overflow
