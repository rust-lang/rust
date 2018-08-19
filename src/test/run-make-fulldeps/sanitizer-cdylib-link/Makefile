-include ../tools.mk

LOG := $(TMPDIR)/log.txt

# This test builds a shared object, then an executable that links it as a native
# rust library (constrast to an rlib). The shared library and executable both
# are compiled with address sanitizer, and we assert that a fault in the cdylib
# is correctly detected.

ifeq ($(TARGET),x86_64-unknown-linux-gnu)
ASAN_SUPPORT=$(SANITIZER_SUPPORT)

# See comment in sanitizer-address/Makefile for why this is here
EXTRA_RUSTFLAG=-C relocation-model=dynamic-no-pic
endif

all:
ifeq ($(ASAN_SUPPORT),1)
	$(RUSTC) -g -Z sanitizer=address --crate-type cdylib --target $(TARGET) $(EXTRA_RUSTFLAG) library.rs
	$(RUSTC) -g -Z sanitizer=address --crate-type bin --target $(TARGET) $(EXTRA_RUSTFLAG) -llibrary program.rs
	LD_LIBRARY_PATH=$(TMPDIR) $(TMPDIR)/program 2>&1 | $(CGREP) stack-buffer-overflow
endif
