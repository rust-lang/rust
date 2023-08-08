# only-windows
# needs-rust-lld

include ../tools.mk

# Ensure that LLD can link
all:
	$(RUSTC) -C linker=rust-lld foo.rs
