# only-windows
# needs-rust-lld

include ../tools.mk

all: foo bar

# Ensure that LLD can link when an .rlib contains a synthetic object
# file referencing exported or used symbols.
foo:
	$(RUSTC) -C linker=rust-lld foo.rs

# Ensure that LLD can link when /WHOLEARCHIVE: is used with an .rlib.
# Previously, lib.rmeta was not marked as (trivially) SAFESEH-aware.
bar: baz
	$(RUSTC) -C linker=rust-lld -C link-arg=/WHOLEARCHIVE:libbaz.rlib bar.rs

baz:
	$(RUSTC) baz.rs
