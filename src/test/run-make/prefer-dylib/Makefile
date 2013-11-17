-include ../tools.mk

all:
	$(RUSTC) bar.rs --dylib --rlib
	$(RUSTC) foo.rs -Z prefer-dynamic
	$(call RUN,foo)
	rm $(TMPDIR)/*bar*
	$(call FAILS,foo)
