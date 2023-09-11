include ../tools.mk

# Verify that the impl_* symbols are preserved. #108030
# only-x86_64-unknown-linux-gnu
# min-llvm-version: 17

all:
	$(RUSTC) -Cdebuginfo=0 -Copt-level=3 lib.rs
	$(RUSTC) -Clto=fat -Cdebuginfo=0 -Copt-level=3 main.rs
