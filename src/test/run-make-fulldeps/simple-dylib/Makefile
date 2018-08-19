-include ../tools.mk
all:
	$(RUSTC) bar.rs --crate-type=dylib -C prefer-dynamic
	$(RUSTC) foo.rs
	$(call RUN,foo)
