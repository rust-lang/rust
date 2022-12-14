include ../../run-make-fulldeps/tools.mk

# only-wasm32-bare

all:
	$(RUSTC) foo.rs --target wasm32-unknown-unknown
	$(NODE) verify-imports.js $(TMPDIR)/foo.wasm a/foo b/foo
	$(RUSTC) foo.rs --target wasm32-unknown-unknown -C lto
	$(NODE) verify-imports.js $(TMPDIR)/foo.wasm a/foo b/foo
	$(RUSTC) foo.rs --target wasm32-unknown-unknown -O
	$(NODE) verify-imports.js $(TMPDIR)/foo.wasm a/foo b/foo
	$(RUSTC) foo.rs --target wasm32-unknown-unknown -O -C lto
	$(NODE) verify-imports.js $(TMPDIR)/foo.wasm a/foo b/foo

	$(RUSTC) bar.rs --target wasm32-unknown-unknown
	$(NODE) verify-imports.js $(TMPDIR)/bar.wasm m1/f m1/g m2/f
	$(RUSTC) bar.rs --target wasm32-unknown-unknown -C lto
	$(NODE) verify-imports.js $(TMPDIR)/bar.wasm m1/f m1/g m2/f
	$(RUSTC) bar.rs --target wasm32-unknown-unknown -O
	$(NODE) verify-imports.js $(TMPDIR)/bar.wasm m1/f m1/g m2/f
	$(RUSTC) bar.rs --target wasm32-unknown-unknown -O -C lto
	$(NODE) verify-imports.js $(TMPDIR)/bar.wasm m1/f m1/g m2/f

	$(RUSTC) baz.rs --target wasm32-unknown-unknown
	$(NODE) verify-imports.js $(TMPDIR)/baz.wasm sqlite/allocate sqlite/deallocate

	$(RUSTC) log.rs --target wasm32-unknown-unknown
	$(NODE) verify-imports.js $(TMPDIR)/log.wasm test/log
