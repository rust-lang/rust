-include ../tools.mk

# Make sure we don't ICE if the linker prints a non-UTF-8 error message.

# The zzz it to allow humans to tab complete or glob this thing.
bad_dir := $(TMPDIR)/zzz$$'\xff'

all:
	$(RUSTC) library.rs
	mkdir $(bad_dir)
	mv $(call STATICLIB,library) $(bad_dir)
	LIBRARY_PATH=$(bad_dir) $(RUSTC) exec.rs 2>&1 | grep this_symbol_not_defined
