include ../tools.mk

# only-x86_64-unknown-linux-gnu

all:
	$(RUSTC) main.rs -o $(TMPDIR)/main
	[ "$$("$(LLVM_BIN_DIR)"/llvm-nm -U $(TMPDIR)/main | grep -c __fixunssfti)" -eq "0" ]
