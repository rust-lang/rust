-include ../tools.mk

all: $(call STATICLIB,cfoo)
	$(RUSTC) foo.rs
	$(RUSTC) bar.rs
	rm $(TMPDIR)/$(call STATICLIB_GLOB,cfoo)
	$(call RUN,bar)
	rm $(TMPDIR)/$(call DYLIB_GLOB,foo)
	$(call FAIL,bar)
