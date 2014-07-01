# This test ensures that if you have the same rlib or dylib at two locations
# in the same path that you don't hit an assertion in the compiler.
#
# Note that this relies on `liburl` to be in the path somewhere else,
# and then our aux-built libraries will collide with liburl (they have
# the same version listed)

-include ../tools.mk

all:
	mkdir $(TMPDIR)/other
	$(RUSTC) foo.rs --crate-type=dylib
	$(RUSTC) foo.rs --crate-type=dylib -o $(TMPDIR)/other/libfoo.so
	$(RUSTC) bar.rs -L $(TMPDIR)/other 2>&1 | \
		grep "multiple dylib candidates"
	rm -rf $(TMPDIR)
	mkdir -p $(TMPDIR)/other
	$(RUSTC) foo.rs --crate-type=rlib
	$(RUSTC) foo.rs --crate-type=rlib -o $(TMPDIR)/other/libfoo.rlib
	$(RUSTC) bar.rs -L $(TMPDIR)/other 2>&1 | \
		grep "multiple rlib candidates"
