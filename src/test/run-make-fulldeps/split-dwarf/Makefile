-include ../tools.mk

# only-linux

all: packed remapped

remapped:
	$(RUSTC) -Z unstable-options -C split-debuginfo=packed -C debuginfo=2 --remap-path-prefix $(TMPDIR)=/a foo.rs -g
	objdump -Wi $(TMPDIR)/foo | grep DW_AT_GNU_dwo_name | (! grep $(TMPDIR)) || exit 1

	$(RUSTC) -Z unstable-options -C split-debuginfo=unpacked -C debuginfo=2 --remap-path-prefix $(TMPDIR)=/a foo.rs -g
	objdump -Wi $(TMPDIR)/foo | grep DW_AT_GNU_dwo_name | (! grep $(TMPDIR)) || exit 1

packed:
	$(RUSTC) -Z unstable-options -C split-debuginfo=packed -C debuginfo=2 foo.rs -g
	rm $(TMPDIR)/foo.dwp
	rm $(TMPDIR)/$(call BIN,foo)
