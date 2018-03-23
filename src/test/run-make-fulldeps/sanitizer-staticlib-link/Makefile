-include ../tools.mk

# This test builds a staticlib, then an executable that links to it.
# The staticlib and executable both  are compiled with address sanitizer, 
# and we assert that a fault in the staticlib is correctly detected.

ifeq ($(TARGET),x86_64-unknown-linux-gnu)
ASAN_SUPPORT=$(SANITIZER_SUPPORT)
EXTRA_RUSTFLAG=
endif

all:
ifeq ($(ASAN_SUPPORT),1)
	$(RUSTC) -g -Z sanitizer=address --crate-type staticlib --target $(TARGET) library.rs
	$(CC) program.c $(call STATICLIB,library) $(call OUT_EXE,program) $(EXTRACFLAGS) $(EXTRACXXFLAGS)
	LD_LIBRARY_PATH=$(TMPDIR) $(TMPDIR)/program 2>&1 | $(CGREP) stack-buffer-overflow
endif

