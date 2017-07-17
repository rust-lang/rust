-include ../tools.mk

# This test builds a shared object, then an executable that links it as a native
# rust library (constrast to an rlib). The shared library and executable both
# are compiled with address sanitizer, and we assert that a fault in the dylib
# is correctly detected.

ifeq ($(TARGET),x86_64-unknown-linux-gnu)
ASAN_SUPPORT=$(SANITIZER_SUPPORT)
EXTRA_RUSTFLAG=
endif

all:
ifeq ($(ASAN_SUPPORT),1)
	$(RUSTC) -g -Z sanitizer=address --crate-type dylib --target $(TARGET) library.rs
	$(RUSTC) -g -Z sanitizer=address --crate-type bin --target $(TARGET) -llibrary program.rs
	LD_LIBRARY_PATH=$(TMPDIR) $(TMPDIR)/program 2>&1 | grep -q stack-buffer-overflow
endif

