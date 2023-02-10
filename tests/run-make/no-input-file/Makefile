include ../../run-make-fulldeps/tools.mk

all:
	$(RUSTC) --print crate-name 2>&1 | diff - no-input-file.stderr
