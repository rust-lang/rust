# only-windows
# needs-rust-lld

include ../../run-make-fulldeps/tools.mk

# Ensure that LLD can link
all:
	$(RUSTC) -C linker=rust-lld foo.rs
