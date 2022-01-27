-include ../../run-make-fulldeps/tools.mk

all:
	$(RUSTDOC) -W help 2>&1 | diff - output-default.stdout
