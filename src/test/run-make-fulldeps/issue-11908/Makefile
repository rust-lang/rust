# This test ensures that if you have the same rlib or dylib at two locations
# in the same path that you don't hit an assertion in the compiler.
#
# Note that this relies on `liburl` to be in the path somewhere else,
# and then our aux-built libraries will collide with liburl (they have
# the same version listed)

-include ../tools.mk

all:
	mkdir $(TMPDIR)/other
	$(RUSTC) foo.rs --crate-type=dylib -C prefer-dynamic
	mv $(call DYLIB,foo) $(TMPDIR)/other
	$(RUSTC) foo.rs --crate-type=dylib -C prefer-dynamic
	$(RUSTC) bar.rs -L $(TMPDIR)/other
	rm -rf $(TMPDIR)
	mkdir -p $(TMPDIR)/other
	$(RUSTC) foo.rs --crate-type=rlib
	mv $(TMPDIR)/libfoo.rlib $(TMPDIR)/other
	$(RUSTC) foo.rs --crate-type=rlib
	$(RUSTC) bar.rs -L $(TMPDIR)/other
