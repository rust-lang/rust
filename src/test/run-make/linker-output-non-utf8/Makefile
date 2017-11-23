-include ../tools.mk

# Make sure we don't ICE if the linker prints a non-UTF-8 error message.

ifdef IS_WINDOWS
# ignore windows

# This does not work in its current form on windows, possibly due to
# gcc bugs or something about valid Windows paths.  See issue #29151
# for more information.
all:

else

# The zzz it to allow humans to tab complete or glob this thing.
bad_dir := $(TMPDIR)/zzz$$'\xff'

all:
	$(RUSTC) library.rs
	mkdir $(bad_dir)
	mv $(TMPDIR)/liblibrary.a $(bad_dir)
	LIBRARY_PATH=$(bad_dir) $(RUSTC) exec.rs 2>&1 | $(CGREP) this_symbol_not_defined

endif
