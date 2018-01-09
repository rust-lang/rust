-include ../tools.mk

# Make sure we don't ICE if the linker prints a non-UTF-8 error message.

# Ignore Windows and Apple

# This does not work in its current form on Windows or Apple APFS due
# to their filesystems requiring paths to be valid Unicode.
ifndef IS_WINDOWS
ifneq ($(shell uname),Darwin)

# The zzz it to allow humans to tab complete or glob this thing.
bad_dir := $(TMPDIR)/zzz$$'\xff'

all:
	$(RUSTC) library.rs
	mkdir $(bad_dir)
	mv $(TMPDIR)/liblibrary.a $(bad_dir)
	LIBRARY_PATH=$(bad_dir) $(RUSTC) exec.rs 2>&1 | $(CGREP) this_symbol_not_defined
else
all:

endif

else
all:

endif
