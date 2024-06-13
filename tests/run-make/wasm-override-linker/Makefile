# needs-force-clang-based-tests

# FIXME(#126180): This test doesn't actually run anywhere, because the only
# CI job that sets RUSTBUILD_FORCE_CLANG_BASED_TESTS runs very few tests.

include ../tools.mk

ifeq ($(TARGET),wasm32-unknown-unknown)
all:
	$(RUSTC) foo.rs --crate-type cdylib --target $(TARGET) -C linker=$(CLANG)
else ifeq ($(TARGET),wasm64-unknown-unknown)
all:
	$(RUSTC) foo.rs --crate-type cdylib --target $(TARGET) -C linker=$(CLANG)
else
all:
endif
