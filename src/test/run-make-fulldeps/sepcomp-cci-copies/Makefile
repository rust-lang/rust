-include ../tools.mk

# Check that cross-crate inlined items are inlined in all compilation units
# that refer to them, and not in any other compilation units.
# Note that we have to pass `-C codegen-units=6` because up to two CGUs may be
# created for each source module (see `rustc_mir::monomorphize::partitioning`).

all:
	$(RUSTC) cci_lib.rs
	$(RUSTC) foo.rs --emit=llvm-ir -C codegen-units=6 \
		-Z inline-in-all-cgus
	[ "$$(cat "$(TMPDIR)"/foo.*.ll | grep -c define\ .*cci_fn)" -eq "2" ]
