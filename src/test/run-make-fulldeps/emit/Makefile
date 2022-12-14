include ../tools.mk

all:
	$(RUSTC) -Copt-level=0 --emit=llvm-bc,llvm-ir,asm,obj,link test-24876.rs
	$(RUSTC) -Copt-level=1 --emit=llvm-bc,llvm-ir,asm,obj,link test-24876.rs
	$(RUSTC) -Copt-level=2 --emit=llvm-bc,llvm-ir,asm,obj,link test-24876.rs
	$(RUSTC) -Copt-level=3 --emit=llvm-bc,llvm-ir,asm,obj,link test-24876.rs
	$(RUSTC) -Copt-level=s --emit=llvm-bc,llvm-ir,asm,obj,link test-24876.rs
	$(RUSTC) -Copt-level=z --emit=llvm-bc,llvm-ir,asm,obj,link test-24876.rs
	$(RUSTC) -Copt-level=0 --emit=llvm-bc,llvm-ir,asm,obj,link test-26235.rs
	$(call RUN,test-26235) || exit 1
	$(RUSTC) -Copt-level=1 --emit=llvm-bc,llvm-ir,asm,obj,link test-26235.rs
	$(call RUN,test-26235) || exit 1
	$(RUSTC) -Copt-level=2 --emit=llvm-bc,llvm-ir,asm,obj,link test-26235.rs
	$(call RUN,test-26235) || exit 1
	$(RUSTC) -Copt-level=3 --emit=llvm-bc,llvm-ir,asm,obj,link test-26235.rs
	$(call RUN,test-26235) || exit 1
	$(RUSTC) -Copt-level=s --emit=llvm-bc,llvm-ir,asm,obj,link test-26235.rs
	$(call RUN,test-26235) || exit 1
	$(RUSTC) -Copt-level=z --emit=llvm-bc,llvm-ir,asm,obj,link test-26235.rs
	$(call RUN,test-26235) || exit 1
