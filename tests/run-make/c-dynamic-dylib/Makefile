# This test checks that dynamic Rust linking with C does not encounter any errors, with dynamic dependencies given preference over static.
# See https://github.com/rust-lang/rust/issues/10434

# ignore-cross-compile
include ../tools.mk

# ignore-apple
#
# This hits an assertion in the linker on older versions of osx apparently

all: $(call DYLIB,cfoo)
	$(RUSTC) foo.rs -C prefer-dynamic
	$(RUSTC) bar.rs
	$(call RUN,bar)
	$(call REMOVE_DYLIBS,cfoo)
	$(call FAIL,bar)
