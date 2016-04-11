-include ../tools.mk
all:
	$(RUSTC) foo.rs --target=my-awesome-platform.json --crate-type=lib --emit=asm
	grep -q -v morestack < $(TMPDIR)/foo.s
	$(RUSTC) foo.rs --target=my-invalid-platform.json 2>&1 | grep -q "Error loading target specification"
	$(RUSTC) foo.rs --target=my-incomplete-platform.json 2>&1 | grep 'Field llvm-target'
	RUST_TARGET_PATH=. $(RUSTC) foo.rs --target=my-awesome-platform --crate-type=lib --emit=asm
	RUST_TARGET_PATH=. $(RUSTC) foo.rs --target=x86_64-unknown-linux-gnu --crate-type=lib --emit=asm
	$(RUSTC) -Z unstable-options --target=my-awesome-platform.json --print target-spec-json > $(TMPDIR)/test-platform.json && $(RUSTC) -Z unstable-options --target=$(TMPDIR)/test-platform.json --print target-spec-json | diff -q $(TMPDIR)/test-platform.json -
