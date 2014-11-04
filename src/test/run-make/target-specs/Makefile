-include ../tools.mk
all:
	$(RUSTC) foo.rs --target=my-awesome-platform.json --crate-type=lib --emit=asm
	grep --quiet --invert-match morestack < $(TMPDIR)/foo.s
	$(RUSTC) foo.rs --target=my-invalid-platform.json 2>&1 | grep --quiet --invert-match "Error loading taget specification"
	$(RUSTC) foo.rs --target=my-incomplete-platform.json 2>&1 | grep 'Field llvm-target'
	RUST_TARGET_PATH=. $(RUSTC) foo.rs --target=my-awesome-platform --crate-type=lib --emit=asm
	RUST_TARGET_PATH=. $(RUSTC) foo.rs --target=x86_64-unknown-linux-gnu --crate-type=lib --emit=asm
	# The built-in target *should* override the one we have here, and thus we
	# should have morestack
	grep --quiet morestack < $(TMPDIR)/foo.s
