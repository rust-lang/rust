-include ../tools.mk

# Test that if we build `b` against a version of `a` that has one set
# of types, it will not run with a dylib that has a different set of
# types.

# NOTE(eddyb) this test only works with the `legacy` mangling,
# and will probably get removed once `legacy` is gone.

all:
	$(RUSTC) a.rs --cfg x -C prefer-dynamic -Z symbol-mangling-version=legacy
	$(RUSTC) b.rs -C prefer-dynamic -Z symbol-mangling-version=legacy
	$(call RUN,b)
	$(RUSTC) a.rs --cfg y -C prefer-dynamic -Z symbol-mangling-version=legacy
	$(call FAIL,b)
