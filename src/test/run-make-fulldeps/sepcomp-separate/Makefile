include ../tools.mk

# Test that separate compilation actually puts code into separate compilation
# units.  `foo.rs` defines `magic_fn` in three different modules, which should
# wind up in three different compilation units.

all:
	$(RUSTC) foo.rs --emit=llvm-ir -C codegen-units=3
	[ "$$(cat "$(TMPDIR)"/foo.*.ll | grep -c define\ .*magic_fn)" -eq "3" ]
