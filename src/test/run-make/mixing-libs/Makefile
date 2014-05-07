-include ../tools.mk

all:
	$(RUSTC) rlib.rs
	$(RUSTC) dylib.rs
	$(RUSTC) rlib.rs --crate-type=dylib
	$(RUSTC) dylib.rs
	rm $(call DYLIB,rlib-*)
	$(RUSTC) prog.rs && exit 1 || exit 0
