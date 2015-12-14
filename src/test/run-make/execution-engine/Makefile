-include ../tools.mk

# FIXME: ignore freebsd
# This is a basic test of LLVM ExecutionEngine functionality using compiled
# Rust code built using the `rustc` crate.

ifeq ($(filter executionengine,$(LLVM_COMPONENTS)),executionengine)

ifneq ($(shell uname),FreeBSD)
all:
	$(RUSTC) test.rs
	$(call RUN,test $(RUSTC))
else
all:

endif

else
all:

endif
