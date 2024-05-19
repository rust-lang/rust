# needs-matching-clang

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
