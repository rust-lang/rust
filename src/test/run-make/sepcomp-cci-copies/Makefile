-include ../tools.mk

# Check that cross-crate inlined items are inlined in all compilation units
# that refer to them, and not in any other compilation units.

all:
	$(RUSTC) cci_lib.rs
	$(RUSTC) foo.rs --emit=ir -C codegen-units=3
	[ "$$(cat "$(TMPDIR)"/foo.?.ll | grep -c define\ .*cci_fn)" -eq "2" ]
