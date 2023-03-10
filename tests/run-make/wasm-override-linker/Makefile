include ../../run-make-fulldeps/tools.mk

ifeq ($(TARGET),wasm32-unknown-unknown)
all:
	$(RUSTC) foo.rs --crate-type cdylib --target $(TARGET) -C linker=clang
else ifeq ($(TARGET),wasm64-unknown-unknown)
all:
	$(RUSTC) foo.rs --crate-type cdylib --target $(TARGET) -C linker=clang
else
all:
endif
