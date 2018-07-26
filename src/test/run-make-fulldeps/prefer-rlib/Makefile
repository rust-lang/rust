-include ../tools.mk

all:
	$(RUSTC) bar.rs --crate-type=dylib --crate-type=rlib
	ls $(TMPDIR)/$(call RLIB_GLOB,bar)
	$(RUSTC) foo.rs
	rm $(TMPDIR)/*bar*
	$(call RUN,foo)
