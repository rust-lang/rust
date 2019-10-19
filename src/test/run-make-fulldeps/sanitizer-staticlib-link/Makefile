# needs-sanitizer-support
# only-x86_64
# only-linux

-include ../tools.mk

# This test builds a staticlib, then an executable that links to it.
# The staticlib and executable both  are compiled with address sanitizer,
# and we assert that a fault in the staticlib is correctly detected.

all:
	$(RUSTC) -g -Z sanitizer=address --crate-type staticlib --target $(TARGET) library.rs
	# Verify that linking without sanitizer fails, i.e., runtime is not part of staticlib:
	$(RUSTC) -g --crate-type bin --target $(TARGET) -L . program.rs && exit 1 || true
	# Verify that linking with sanitizer succeeds and detects overflow in library:
	$(RUSTC) -g -Z sanitizer=address --crate-type bin --target $(TARGET) -L . program.rs
	$(TMPDIR)/program 2>&1 | $(CGREP) stack-buffer-overflow

