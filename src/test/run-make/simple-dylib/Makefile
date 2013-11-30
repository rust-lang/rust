-include ../tools.mk
all:
	$(RUSTC) bar.rs --dylib
	$(RUSTC) foo.rs
	$(call RUN,foo)
