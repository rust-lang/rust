-include ../../run-make-fulldeps/tools.mk

# NOTE we use --emit=llvm-ir to avoid running the linker (linking will fail because there's no main
# in this crate)
all:
	$(RUSTC) panic-impl-provider.rs
	$(RUSTC) panic-impl-consumer.rs -C panic=abort --emit=llvm-ir -L $(TMPDIR)
