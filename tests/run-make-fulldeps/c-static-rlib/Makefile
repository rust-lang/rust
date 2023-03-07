include ../tools.mk

all: $(call NATIVE_STATICLIB,cfoo)
	$(RUSTC) foo.rs
	$(RUSTC) bar.rs
	$(call REMOVE_RLIBS,foo)
	rm $(call NATIVE_STATICLIB,cfoo)
	$(call RUN,bar)
