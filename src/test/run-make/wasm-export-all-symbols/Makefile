-include ../../run-make-fulldeps/tools.mk

ifeq ($(TARGET),wasm32-unknown-unknown)
all:
	$(RUSTC) bar.rs --target wasm32-unknown-unknown
	$(RUSTC) foo.rs --target wasm32-unknown-unknown
	$(NODE) verify.js $(TMPDIR)/foo.wasm
	$(RUSTC) bar.rs --target wasm32-unknown-unknown -O
	$(RUSTC) foo.rs --target wasm32-unknown-unknown -O
	$(NODE) verify.js $(TMPDIR)/foo.wasm
	$(RUSTC) foo.rs --target wasm32-unknown-unknown -C lto
	$(NODE) verify.js $(TMPDIR)/foo.wasm
else
all:
endif

