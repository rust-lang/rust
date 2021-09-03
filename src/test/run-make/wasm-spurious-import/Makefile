-include ../../run-make-fulldeps/tools.mk

# only-wasm32-bare

all:
	$(RUSTC) main.rs -C overflow-checks=yes -C panic=abort -C lto -C opt-level=z --target wasm32-unknown-unknown
	$(NODE) verify.js $(TMPDIR)/main.wasm
