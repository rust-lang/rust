-include ../tools.mk

# only-linux

all:
	$(RUSTC) -Z split-dwarf=split foo.rs
	rm $(TMPDIR)/foo.dwp
	rm $(TMPDIR)/$(call BIN,foo)
