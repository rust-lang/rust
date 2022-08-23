include ../tools.mk

all: $(call NATIVE_STATICLIB,cfoo)
	$(RUSTC) foo.rs -C prefer-dynamic
	$(RUSTC) bar.rs
	rm $(call NATIVE_STATICLIB,cfoo)
	$(call RUN,bar)
	$(call REMOVE_DYLIBS,foo)
	$(call FAIL,bar)
