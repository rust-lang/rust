-include ../tools.mk

all: $(call STATICLIB,cfoo)
	$(RUSTC) foo.rs
	$(RUSTC) bar.rs
	rm $(TMPDIR)/$(call STATICLIB_GLOB,cfoo)
	$(call RUN,bar)
	$(call REMOVE_DYLIBS,foo)
	$(call FAIL,bar)
