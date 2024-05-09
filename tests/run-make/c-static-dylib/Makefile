# This test checks that static Rust linking with C does not encounter any errors, with dynamic dependencies given preference over static.
# See https://github.com/rust-lang/rust/issues/10434

# ignore-cross-compile
include ../tools.mk

all: $(call NATIVE_STATICLIB,cfoo)
	$(RUSTC) foo.rs -C prefer-dynamic
	$(RUSTC) bar.rs
	rm $(call NATIVE_STATICLIB,cfoo)
	$(call RUN,bar)
	$(call REMOVE_DYLIBS,foo)
	$(call FAIL,bar)
