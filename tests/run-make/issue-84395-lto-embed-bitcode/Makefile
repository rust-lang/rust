# needs-force-clang-based-tests

# FIXME(#126180): This test doesn't actually run anywhere, because the only
# CI job that sets RUSTBUILD_FORCE_CLANG_BASED_TESTS runs very few tests.

# This test makes sure the embed bitcode in elf created with
# lto-embed-bitcode=optimized is valid llvm BC module.

include ../tools.mk

all:
	$(RUSTC) test.rs --target $(TARGET) -Clink-arg=-fuse-ld=lld -Clinker-plugin-lto -Clinker=$(CLANG) -Clink-arg=-Wl,--plugin-opt=-lto-embed-bitcode=optimized -Zemit-thin-lto=no
	$(LLVM_BIN_DIR)/objcopy --dump-section .llvmbc=$(TMPDIR)/test.bc $(TMPDIR)/test
	$(LLVM_BIN_DIR)/llvm-dis $(TMPDIR)/test.bc
