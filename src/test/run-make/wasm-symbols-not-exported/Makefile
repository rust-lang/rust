include ../../run-make-fulldeps/tools.mk

# only-wasm32-bare

all:
	$(RUSTC) foo.rs --target wasm32-unknown-unknown
	$(NODE) verify-exported-symbols.js $(TMPDIR)/foo.wasm
	$(RUSTC) foo.rs --target wasm32-unknown-unknown -O
	$(NODE) verify-exported-symbols.js $(TMPDIR)/foo.wasm
	$(RUSTC) bar.rs --target wasm32-unknown-unknown
	$(NODE) verify-exported-symbols.js $(TMPDIR)/bar.wasm
	$(RUSTC) bar.rs --target wasm32-unknown-unknown -O
	$(NODE) verify-exported-symbols.js $(TMPDIR)/bar.wasm
