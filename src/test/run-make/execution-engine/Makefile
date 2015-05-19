-include ../tools.mk

# This is a basic test of LLVM ExecutionEngine functionality using compiled
# Rust code built using the `rustc` crate.

all:
	$(RUSTC) test.rs
	$(call RUN,test $(RUSTC))
