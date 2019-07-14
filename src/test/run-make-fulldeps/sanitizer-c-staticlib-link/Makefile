# needs-sanitizer-support
# only-x86_64
-include ../tools.mk

ifeq ($(TARGET),x86_64-apple-darwin)
# sanitizers are always built as dylibs on osx
EXTRACFLAGS=-L$(TARGET_RPATH_DIR) -l__rustc__clang_rt.asan_osx_dynamic
LD_LIBRARY_PATH="$(TARGET_RPATH_DIR):$(TMPDIR)"
else
LD_LIBRARY_PATH=$(TMPDIR)
endif

# This test builds a staticlib, then an executable that links to it.
# The staticlib is compiled with address sanitizer, and we assert that a fault
# in the staticlib is correctly detected.

all:
	$(RUSTC) -g -Z sanitizer=address --crate-type staticlib --target $(TARGET) library.rs
	$(CC) program.c $(call STATICLIB,library) $(call OUT_EXE,program) $(EXTRACFLAGS) $(EXTRACXXFLAGS)
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) $(TMPDIR)/program 2>&1 | $(CGREP) stack-buffer-overflow
