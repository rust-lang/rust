-include ../tools.mk

# Test that #[inline(always)] functions still get inlined across compilation
# unit boundaries.  Compilation should produce three IR files, with each one
# containing a definition of the inlined function.  Also, the non-#[inline]
# function should be defined in only one compilation unit.

all:
	$(RUSTC) foo.rs --emit=ir -C codegen-units=3
	[ "$$(cat "$(TMPDIR)"/foo.?.ll | grep -c define\ i32\ .*inlined)" -eq "1" ]
	[ "$$(cat "$(TMPDIR)"/foo.?.ll | grep -c define\ available_externally\ i32\ .*inlined)" -eq "2" ]
	[ "$$(cat "$(TMPDIR)"/foo.?.ll | grep -c define\ i32\ .*normal)" -eq "1" ]
	[ "$$(cat "$(TMPDIR)"/foo.?.ll | grep -c declare\ i32\ .*normal)" -eq "2" ]
