include ../tools.mk

# rust-lang/rust#70924: Test that if we add rust-src component in between two
# incremental compiles, the compiler does not ICE on the second.

# This test uses `ln -s` rather than copying to save testing time, but its
# usage doesn't work on windows. So ignore windows.

# ignore-windows

SYSROOT:=$(shell $(RUSTC) --print sysroot)
FAKEROOT=$(TMPDIR)/fakeroot
INCR=$(TMPDIR)/incr

# Make a local copy of the sysroot; then remove the rust-src part of it, if
# present, for the *first* build. Then put in a facsimile of the rust-src
# component for the second build, in order to expose the ICE from issue #70924.
#
# Note that it is much easier to just do `cp -a $(SYSROOT)/* $(FAKEROOT)` as a
# first step, but I am concerned that would be too expensive in a unit test
# compared to making symbolic links.
#
# Anyway, the pattern you'll see here is: For every prefix in
# root/lib/rustlib/src, link all of prefix parent content, then remove the
# prefix, then loop on the next prefix. This way, we basically create a copy of
# the context around root/lib/rustlib/src, and can freely add/remove the src
# component itself.
all:
	mkdir $(FAKEROOT)
	ln -s $(SYSROOT)/* $(FAKEROOT)
	rm -f $(FAKEROOT)/lib
	mkdir $(FAKEROOT)/lib
	ln -s $(SYSROOT)/lib/* $(FAKEROOT)/lib
	rm -f $(FAKEROOT)/lib/rustlib
	mkdir $(FAKEROOT)/lib/rustlib
	ln -s $(SYSROOT)/lib/rustlib/* $(FAKEROOT)/lib/rustlib
	rm -f $(FAKEROOT)/lib/rustlib/src
	mkdir $(FAKEROOT)/lib/rustlib/src
	ln -s $(SYSROOT)/lib/rustlib/src/* $(FAKEROOT)/lib/rustlib/src
	rm -f $(FAKEROOT)/lib/rustlib/src/rust
	$(RUSTC) --sysroot $(FAKEROOT) -C incremental=$(INCR) main.rs
	mkdir -p $(FAKEROOT)/lib/rustlib/src/rust/src/libstd
	touch $(FAKEROOT)/lib/rustlib/src/rust/src/libstd/lib.rs
	$(RUSTC) --sysroot $(FAKEROOT) -C incremental=$(INCR) main.rs
