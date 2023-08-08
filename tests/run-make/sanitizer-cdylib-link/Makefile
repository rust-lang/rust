# needs-sanitizer-support
# needs-sanitizer-address

include ../tools.mk

LOG := $(TMPDIR)/log.txt

# This test builds a shared object, then an executable that links it as a native
# rust library (contrast to an rlib). The shared library and executable both
# are compiled with address sanitizer, and we assert that a fault in the cdylib
# is correctly detected.

all:
	$(RUSTC) -g -Z sanitizer=address --crate-type cdylib --target $(TARGET) library.rs
	$(RUSTC) -g -Z sanitizer=address --crate-type bin --target $(TARGET) -llibrary program.rs
	LD_LIBRARY_PATH=$(TMPDIR) $(TMPDIR)/program 2>&1 | $(CGREP) stack-buffer-overflow
