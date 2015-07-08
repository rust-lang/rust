-include ../tools.mk

# FIXME: ignore freebsd
# This is a basic test of LLVM ExecutionEngine functionality using compiled
# Rust code built using the `rustc` crate.

ifneq ($(shell uname),FreeBSD)
all:
	$(RUSTC) test.rs
	$(call RUN,test $(RUSTC))
else
all:

endif
