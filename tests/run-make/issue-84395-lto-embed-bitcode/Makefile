# needs-matching-clang

# This test makes sure the embed bitcode in elf created with
# lto-embed-bitcode=optimized is valid llvm BC module.

include ../../run-make-fulldeps/tools.mk

all:
	$(RUSTC) test.rs --target $(TARGET) -Clink-arg=-fuse-ld=lld -Clinker-plugin-lto -Clinker=$(CLANG) -Clink-arg=-Wl,--plugin-opt=-lto-embed-bitcode=optimized -Zemit-thin-lto=no
	$(LLVM_BIN_DIR)/objcopy --dump-section .llvmbc=$(TMPDIR)/test.bc $(TMPDIR)/test
	$(LLVM_BIN_DIR)/llvm-dis $(TMPDIR)/test.bc
