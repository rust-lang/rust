-include ../tools.mk

all:
	$(RUSTC) -v
	$(RUSTC) -v verbose
	$(RUSTC) -v bad_arg && exit 1 || exit 0
	$(RUSTC) --version verbose
	$(RUSTC) --version bad_arg && exit 1 || exit 0
