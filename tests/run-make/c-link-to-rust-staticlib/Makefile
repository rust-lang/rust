# This test checks that C linking with Rust does not encounter any errors, with static libraries.
# See https://github.com/rust-lang/rust/issues/10434

# ignore-cross-compile
include ../tools.mk

# ignore-freebsd
# FIXME

all:
	$(RUSTC) foo.rs
	$(CC) bar.c $(call STATICLIB,foo) $(call OUT_EXE,bar) \
		$(EXTRACFLAGS) $(EXTRACXXFLAGS)
	$(call RUN,bar)
	rm $(call STATICLIB,foo)
	$(call RUN,bar)
