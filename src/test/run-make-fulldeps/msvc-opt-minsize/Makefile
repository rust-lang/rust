-include ../tools.mk

all:
	$(RUSTC) foo.rs -Copt-level=z 2>&1
	$(call RUN,foo)
