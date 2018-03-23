-include ../tools.mk

all:
	$(RUSTC) bar.rs --crate-type=dylib --crate-type=rlib -C prefer-dynamic
	$(RUSTC) foo.rs -C prefer-dynamic
	$(call RUN,foo)
	rm $(TMPDIR)/*bar*
	$(call FAIL,foo)
