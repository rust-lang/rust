-include ../tools.mk

all: $(call NATIVE_STATICLIB,foo)
	$(RUSTC) foo.rs
	$(RUSTC) bar.rs
	$(call RUN,bar)
