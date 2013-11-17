-include ../tools.mk

all:
	$(RUSTC) bar.rs --dylib --rlib
	ls $(TMPDIR)/$(call RLIB_GLOB,bar)
	$(RUSTC) foo.rs
	rm $(TMPDIR)/*bar*
	$(call RUN,foo)
