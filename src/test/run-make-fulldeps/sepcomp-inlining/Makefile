include ../tools.mk

# Test that #[inline] functions still get inlined across compilation unit
# boundaries. Compilation should produce three IR files, but only the two
# compilation units that have a usage of the #[inline] function should
# contain a definition. Also, the non-#[inline] function should be defined
# in only one compilation unit.

all:
	$(RUSTC) foo.rs --emit=llvm-ir -C codegen-units=3 \
		-Z inline-in-all-cgus
	[ "$$(cat "$(TMPDIR)"/foo.*.ll | grep -c define\ i32\ .*inlined)" -eq "0" ]
	[ "$$(cat "$(TMPDIR)"/foo.*.ll | grep -c define\ internal\ i32\ .*inlined)" -eq "2" ]
	[ "$$(cat "$(TMPDIR)"/foo.*.ll | grep -c define\ hidden\ i32\ .*normal)" -eq "1" ]
	[ "$$(cat "$(TMPDIR)"/foo.*.ll | grep -c declare\ hidden\ i32\ .*normal)" -eq "2" ]
