-include ../tools.mk

all:
	$(RUSTC) foo.rs -C target-cpu=native
	$(call RUN,foo)
