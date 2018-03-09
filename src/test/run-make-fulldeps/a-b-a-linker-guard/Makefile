-include ../tools.mk

# Test that if we build `b` against a version of `a` that has one set
# of types, it will not run with a dylib that has a different set of
# types.

all:
	$(RUSTC) a.rs --cfg x -C prefer-dynamic
	$(RUSTC) b.rs -C prefer-dynamic
	$(call RUN,b)
	$(RUSTC) a.rs --cfg y -C prefer-dynamic
	$(call FAIL,b)
