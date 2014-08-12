-include ../tools.mk

# Test to make sure that inner functions within a polymorphic outer function
# don't get re-translated when the outer function is monomorphized.  The test
# code monomorphizes the outer functions several times, but the magic constants
# used in the inner functions should each appear only once in the generated IR.

all:
	$(RUSTC) foo.rs --emit=ir
	[ "$$(grep -c 8675309 "$(TMPDIR)/foo.ll")" -eq "1" ]
	[ "$$(grep -c 11235813 "$(TMPDIR)/foo.ll")" -eq "1" ]
