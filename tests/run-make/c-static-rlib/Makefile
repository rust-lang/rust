# This test checks that static Rust linking with C does not encounter any errors, with static dependencies given preference over dynamic. (This is the default behaviour.)
# See https://github.com/rust-lang/rust/issues/10434

# ignore-cross-compile
include ../tools.mk

all: $(call NATIVE_STATICLIB,cfoo)
	$(RUSTC) foo.rs
	$(RUSTC) bar.rs
	$(call REMOVE_RLIBS,foo)
	rm $(call NATIVE_STATICLIB,cfoo)
	$(call RUN,bar)
