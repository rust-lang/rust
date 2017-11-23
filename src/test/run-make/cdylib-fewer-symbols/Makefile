# Test that allocator-related symbols don't show up as exported from a cdylib as
# they're internal to Rust and not part of the public ABI.

-include ../tools.mk

ifdef IS_MSVC
all:
	true
else
all:
	$(RUSTC) foo.rs
	nm -g "$(call DYLIB,foo)" | $(CGREP) -v __rdl_ __rde_ __rg_ __rust_
endif
