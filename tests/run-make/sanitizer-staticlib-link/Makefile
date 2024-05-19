# needs-sanitizer-support
# needs-sanitizer-address

include ../tools.mk

# This test first builds a staticlib with AddressSanitizer and checks that
# linking it to an executable fails due to the missing sanitizer runtime.
# It then builds an executable linking to the staticlib and checks that
# the fault in the staticlib is detected correctly.

# Note that checking for the link failure actually checks two things at once:
#   1) That the library has the sanitizer intrumentation
#   2) and that library does not have the sanitizer runtime

all:
	$(RUSTC) -g -Z sanitizer=address --crate-type staticlib --target $(TARGET) library.rs
	! $(CC) program.c $(call STATICLIB,library) $(call OUT_EXE,program) $(EXTRACFLAGS) $(EXTRACXXFLAGS)
	$(RUSTC) -g -Z sanitizer=address --crate-type bin --target $(TARGET) -L . program.rs
	LD_LIBRARY_PATH=$(TMPDIR) $(TMPDIR)/program 2>&1 | $(CGREP) stack-buffer-overflow

