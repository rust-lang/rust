include ../../run-make-fulldeps/tools.mk

DYLIB_NAME := $(shell echo | $(RUSTC) --crate-name foo --crate-type dylib --print file-names -)

all:
	echo >> $(TMPDIR)/$(DYLIB_NAME)
	$(RUSTC) --crate-type lib --extern foo=$(TMPDIR)/$(DYLIB_NAME) bar.rs 2>&1 | $(CGREP) 'invalid metadata files for crate `foo`'
