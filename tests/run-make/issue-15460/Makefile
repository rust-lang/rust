include ../tools.mk

all: $(call NATIVE_STATICLIB,foo)
	$(RUSTC) foo.rs -C extra-filename=-383hf8 -C prefer-dynamic
	$(RUSTC) bar.rs
	$(call RUN,bar)
