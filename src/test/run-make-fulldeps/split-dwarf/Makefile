-include ../tools.mk

# only-linux

all: packed remapped

remapped:
	$(RUSTC) -Z unstable-options -C split-debuginfo=packed -C debuginfo=2 --remap-path-prefix $$PWD=/a foo.rs -g
	objdump -Wi $(TMPDIR)/foo | grep $$PWD && exit 1 || exit 0

	$(RUSTC) -Z unstable-options -C split-debuginfo=unpacked -C debuginfo=2 --remap-path-prefix $$PWD=/a foo.rs -g
	objdump -Wi $(TMPDIR)/foo | grep $$PWD && exit 1 || exit 0

packed:
	$(RUSTC) -Z unstable-options -C split-debuginfo=packed -C debuginfo=2 foo.rs -g
	rm $(TMPDIR)/foo.dwp
	rm $(TMPDIR)/$(call BIN,foo)
