-include ../tools.mk

# only-linux
# min-llvm-version: 11.0

all:
	$(RUSTC) -Z unstable-options -C split-debuginfo=packed foo.rs -g
	rm $(TMPDIR)/foo.dwp
	rm $(TMPDIR)/$(call BIN,foo)
