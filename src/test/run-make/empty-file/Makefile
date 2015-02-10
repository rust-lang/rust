-include ../tools.mk

all:
	$(RUSTC) --emit=asm,llvm-bc,llvm-ir,obj,dep-info empty.rs
	$(RUSTC) --emit=link --crate-type=rlib,dylib,staticlib empty.rs
