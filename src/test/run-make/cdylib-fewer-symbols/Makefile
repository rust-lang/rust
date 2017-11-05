# Test that allocator-related symbols don't show up as exported from a cdylib as
# they're internal to Rust and not part of the public ABI.

-include ../tools.mk

ifdef IS_MSVC
all:
	true
else
all:
	$(RUSTC) foo.rs
	nm -g "$(call DYLIB,foo)"
	nm -g "$(call DYLIB,foo)" | grep -vq __rdl_
	nm -g "$(call DYLIB,foo)" | grep -vq __rde_
	nm -g "$(call DYLIB,foo)" | grep -vq __rg_
	nm -g "$(call DYLIB,foo)" | grep -vq __rust_
endif
