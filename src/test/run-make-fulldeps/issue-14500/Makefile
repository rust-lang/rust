-include ../tools.mk

# Test to make sure that reachable extern fns are always available in final
# productcs, including when LTO is used. In this test, the `foo` crate has a
# reahable symbol, and is a dependency of the `bar` crate. When the `bar` crate
# is compiled with LTO, it shouldn't strip the symbol from `foo`, and that's the
# only way that `foo.c` will successfully compile.

all:
	$(RUSTC) foo.rs --crate-type=rlib
	$(RUSTC) bar.rs --crate-type=staticlib -C lto -L. -o $(TMPDIR)/libbar.a
	$(CC) foo.c $(TMPDIR)/libbar.a $(EXTRACFLAGS) $(call OUT_EXE,foo)
	$(call RUN,foo)
