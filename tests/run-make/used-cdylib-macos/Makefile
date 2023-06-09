include ../tools.mk

# only-macos
#
# This checks that `#[used]` passes through to the linker on
# darwin. This is subject to change in the future, see
# https://github.com/rust-lang/rust/pull/93718 for discussion

all:
	$(RUSTC) -Copt-level=3 dylib_used.rs
	nm $(TMPDIR)/libdylib_used.dylib | $(CGREP) VERY_IMPORTANT_SYMBOL
