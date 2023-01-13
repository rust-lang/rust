# Everyone uses make for building Rust

foo: bar.rlib
	$(RUSTC) --crate-type bin --extern bar=bar.rlib

%.rlib: %.rs
	$(RUSTC) --crate-type lib $<
