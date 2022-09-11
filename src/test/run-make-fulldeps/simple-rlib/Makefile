include ../tools.mk
all:
	$(RUSTC) bar.rs --crate-type=rlib
	$(RUSTC) foo.rs
	$(call RUN,foo)
