include ../../run-make-fulldeps/tools.mk

all:
	echo 'fn main() {}' | $(BARE_RUSTC) - --out-dir=$(TMPDIR)/random_directory_that_does_not_exist_ir/ --emit=llvm-ir
	echo 'fn main() {}' | $(BARE_RUSTC) - --out-dir=$(TMPDIR)/random_directory_that_does_not_exist_bc/ --emit=llvm-bc
