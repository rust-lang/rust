-include ../tools.mk

all: $(call STATICLIB,cfoo)
	$(RUSTC) foo.rs
	$(RUSTC) bar.rs
	$(call REMOVE_RLIBS,foo)
	rm $(TMPDIR)/$(call STATICLIB_GLOB,cfoo)
	$(call RUN,bar)
