# This test verifies that "-Z trace-macros"  works as it should. The traditional
# "hello world" program provides a small example of this as not only println! is
# listed, but also print! (since println! expands to this)

-include ../tools.mk

all:
	$(RUSTC) -Z trace-macros hello.rs > $(TMPDIR)/hello.out
	diff -u $(TMPDIR)/hello.out hello.trace
