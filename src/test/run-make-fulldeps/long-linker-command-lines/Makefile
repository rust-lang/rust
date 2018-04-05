-include ../tools.mk

all:
	$(RUSTC) foo.rs -g -O
	RUSTC="$(RUSTC_ORIGINAL)" $(call RUN,foo)
