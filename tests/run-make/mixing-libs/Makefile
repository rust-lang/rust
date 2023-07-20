# ignore-cross-compile
include ../tools.mk

all:
	$(RUSTC) rlib.rs --crate-type=rlib --crate-type=dylib
	$(RUSTC) dylib.rs # no -Cprefer-dynamic so statically linking librlib.rlib
	$(call REMOVE_DYLIBS,rlib) # remove librlib.so to test that prog.rs doesn't get confused about the removed dylib version of librlib
	$(RUSTC) prog.rs && exit 1 || exit 0
