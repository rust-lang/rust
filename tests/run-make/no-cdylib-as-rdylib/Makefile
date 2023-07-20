# ignore-cross-compile
include ../tools.mk

# Test that rustc will not attempt to link against a cdylib as if
# it is a rust dylib when an rlib for the same crate is available.
# Previously rustc didn't actually check if any further formats of
# a crate which has been loaded are of the same version and if
# they are actually valid. This caused a cdylib to be interpreted
# as rust dylib as soon as the corresponding rlib was loaded. As
# cdylibs don't export any rust symbols, linking would fail if
# rustc decides to link against the cdylib rather than the rlib.

all:
	$(RUSTC) bar.rs --crate-type=rlib --crate-type=cdylib
	$(RUSTC) foo.rs -C prefer-dynamic
	$(call RUN,foo)
