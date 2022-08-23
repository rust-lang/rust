include ../tools.mk

# Test that we don't run into an assertion when using a Rust dylib dependency
# while compiling with full LTO.
# See https://github.com/rust-lang/rust/issues/59137

all:
	$(RUSTC) a_dylib.rs --crate-type=dylib -C prefer-dynamic
	$(RUSTC) main.rs -C lto
	$(call RUN,main)
