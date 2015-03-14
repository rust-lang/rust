-include ../tools.mk

all:
	$(RUSTC) foo.rs
	$(call RUN,foo)
	rm $(TMPDIR)/$(call DYLIB_GLOB,foo)
