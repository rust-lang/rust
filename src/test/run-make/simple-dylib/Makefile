-include ../tools.mk
all:
	$(RUSTC) bar.rs --crate-type=dylib
	$(RUSTC) foo.rs
	$(call RUN,foo)
