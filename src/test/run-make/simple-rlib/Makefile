-include ../tools.mk
all:
	$(RUSTC) bar.rs --rlib
	$(RUSTC) foo.rs
	$(call RUN,foo)
