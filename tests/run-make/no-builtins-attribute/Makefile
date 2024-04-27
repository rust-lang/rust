include ../tools.mk

# We want to check if `no-builtins` is also added to the function declarations in the used crate.

all:
	$(RUSTC) no_builtins.rs --emit=link
	$(RUSTC) main.rs --emit=llvm-ir

	cat "$(TMPDIR)"/main.ll | "$(LLVM_FILECHECK)" filecheck.main.txt
