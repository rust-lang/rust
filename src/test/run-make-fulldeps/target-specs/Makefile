include ../tools.mk
all:
	$(RUSTC) foo.rs --target=my-awesome-platform.json --crate-type=lib --emit=asm
	$(CGREP) -v morestack < $(TMPDIR)/foo.s
	$(RUSTC) foo.rs --target=my-invalid-platform.json 2>&1 | $(CGREP) "Error loading target specification"
	$(RUSTC) foo.rs --target=my-incomplete-platform.json 2>&1 | $(CGREP) 'Field llvm-target'
	RUST_TARGET_PATH=. $(RUSTC) foo.rs --target=my-awesome-platform --crate-type=lib --emit=asm
	RUST_TARGET_PATH=. $(RUSTC) foo.rs --target=my-x86_64-unknown-linux-gnu-platform --crate-type=lib --emit=asm
	$(RUSTC) -Z unstable-options --target=my-awesome-platform.json --print target-spec-json > $(TMPDIR)/test-platform.json && $(RUSTC) -Z unstable-options --target=$(TMPDIR)/test-platform.json --print target-spec-json | diff -q $(TMPDIR)/test-platform.json -
	$(RUSTC) foo.rs --target=definitely-not-builtin-target 2>&1 | $(CGREP) 'may not set is_builtin'
	$(RUSTC) foo.rs --target=mismatching-data-layout --crate-type=lib
