# Exercise unwinding a panic. This catches a panic across an FFI boundary and downcasts it into an integer. The Rust code that panics is in the same directory.
# See https://github.com/rust-lang/rust/commit/baf227ea0c1e07fc54395a51e4b3881d701180cb

# ignore-cross-compile
# needs-unwind
include ../tools.mk

all: $(call NATIVE_STATICLIB,add)
	$(RUSTC) main.rs
	$(call RUN,main) || exit 1
