-include ../../run-make-fulldeps/tools.mk

all:
	$(BARE_RUSTDOC) 2>&1 | diff - output-default.stdout
