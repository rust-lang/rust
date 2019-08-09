# needs-sanitizer-support
# only-x86_64
# only-linux

-include ../tools.mk

LOG := $(TMPDIR)/log.txt

# This test builds a shared object, then an executable that links it as a native
# rust library (contrast to an rlib). The shared library and executable both
# are compiled with address sanitizer, and we assert that a fault in the dylib
# is correctly detected.

# See comment in sanitizer-address/Makefile for why this is here
EXTRA_RUSTFLAG=-C relocation-model=dynamic-no-pic

all:
	$(RUSTC) -g -Z sanitizer=address --crate-type dylib --target $(TARGET) $(EXTRA_RUSTFLAG) library.rs
	$(RUSTC) -g -Z sanitizer=address --crate-type bin --target $(TARGET) $(EXTRA_RUSTFLAG) -llibrary program.rs
	LD_LIBRARY_PATH=$(TMPDIR) $(TMPDIR)/program 2>&1 | $(CGREP) stack-buffer-overflow
