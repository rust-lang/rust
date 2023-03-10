include ../../run-make-fulldeps/tools.mk

# only-wasm32-bare

all:
	$(RUSTC) foo.rs --crate-type cdylib --target wasm32-unknown-unknown -C linker=clang
