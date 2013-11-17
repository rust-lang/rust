-include ../tools.mk

all: $(call STATICLIB,cfoo)
	$(RUSTC) foo.rs
	$(RUSTC) bar.rs
	rm $(TMPDIR)/$(call RLIB_GLOB,foo)
	rm $(TMPDIR)/$(call STATICLIB_GLOB,cfoo)
	$(call RUN,bar)
