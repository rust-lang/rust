-include ../tools.mk

all:
	$(RUSTC) -Copt-level=0 --emit=llvm-bc,llvm-ir,asm,obj,link test.rs
	$(RUSTC) -Copt-level=1 --emit=llvm-bc,llvm-ir,asm,obj,link test.rs
	$(RUSTC) -Copt-level=2 --emit=llvm-bc,llvm-ir,asm,obj,link test.rs
	$(RUSTC) -Copt-level=3 --emit=llvm-bc,llvm-ir,asm,obj,link test.rs
