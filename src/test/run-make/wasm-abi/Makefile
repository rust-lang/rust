include ../../run-make-fulldeps/tools.mk

# only-wasm32-bare

all:
	$(RUSTC) foo.rs --target wasm32-unknown-unknown
	$(NODE) foo.js $(TMPDIR)/foo.wasm
