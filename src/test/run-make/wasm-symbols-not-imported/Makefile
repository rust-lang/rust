-include ../../run-make-fulldeps/tools.mk

ifeq ($(TARGET),wasm32-unknown-unknown)
all:
	$(RUSTC) foo.rs --target wasm32-unknown-unknown
	$(NODE) verify-no-imports.js $(TMPDIR)/foo.wasm
	$(RUSTC) foo.rs --target wasm32-unknown-unknown -C lto
	$(NODE) verify-no-imports.js $(TMPDIR)/foo.wasm
	$(RUSTC) foo.rs --target wasm32-unknown-unknown -O
	$(NODE) verify-no-imports.js $(TMPDIR)/foo.wasm
	$(RUSTC) foo.rs --target wasm32-unknown-unknown -O -C lto
	$(NODE) verify-no-imports.js $(TMPDIR)/foo.wasm
else
all:
endif

