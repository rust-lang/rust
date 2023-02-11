include ../../run-make-fulldeps/tools.mk

# only-wasm32-bare

all:
	$(RUSTC) bar.rs --target wasm32-unknown-unknown
	$(RUSTC) foo.rs --target wasm32-unknown-unknown
	$(NODE) verify.js $(TMPDIR)/foo.wasm
	$(RUSTC) main.rs --target wasm32-unknown-unknown
	$(NODE) verify.js $(TMPDIR)/main.wasm
	$(RUSTC) bar.rs --target wasm32-unknown-unknown -O
	$(RUSTC) foo.rs --target wasm32-unknown-unknown -O
	$(NODE) verify.js $(TMPDIR)/foo.wasm
	$(RUSTC) main.rs --target wasm32-unknown-unknown -O
	$(NODE) verify.js $(TMPDIR)/main.wasm
	$(RUSTC) foo.rs --target wasm32-unknown-unknown -C lto
	$(NODE) verify.js $(TMPDIR)/foo.wasm
	$(RUSTC) main.rs --target wasm32-unknown-unknown -C lto
	$(NODE) verify.js $(TMPDIR)/main.wasm
