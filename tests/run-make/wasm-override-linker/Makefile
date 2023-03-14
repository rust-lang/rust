include ../../run-make-fulldeps/tools.mk

ifeq ($(TARGET),wasm32-unknown-unknown)
all:
	$(RUSTC) foo.rs --crate-type cdylib --target wasm32-unknown-unknown -C linker=clang
else
all:
endif
