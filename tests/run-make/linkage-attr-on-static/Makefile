# ignore-cross-compile
include ../tools.mk

all: $(call NATIVE_STATICLIB,foo)
	$(RUSTC) bar.rs
	$(call RUN,bar) || exit 1
