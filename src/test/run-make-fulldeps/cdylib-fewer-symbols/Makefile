# Test that allocator-related symbols don't show up as exported from a cdylib as
# they're internal to Rust and not part of the public ABI.

-include ../tools.mk

# FIXME: The __rdl_ and __rust_ symbol still remains, no matter using MSVC or GNU
# See https://github.com/rust-lang/rust/pull/46207#issuecomment-347561753
ifdef IS_WINDOWS
all:
	true
else
all:
	$(RUSTC) foo.rs
	nm -g "$(call DYLIB,foo)" | $(CGREP) -v __rdl_ __rde_ __rg_ __rust_
endif
