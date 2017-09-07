-include ../tools.mk

all:
	$(RUSTC) foo.rs -g
	RUSTC="$(RUSTC_ORIGINAL)" $(call RUN,foo)
