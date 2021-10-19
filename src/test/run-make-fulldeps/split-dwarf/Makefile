-include ../tools.mk

# only-linux

# Needs commit 6ffcb2937c96bd0d7a55b984b5eb8f381b68e322,
# "[llvm-dwp] Join dwo paths correctly when DWOPath is absolute",
# backported to 12.x in b1106a5b3bc94f6da11682007d101823f81bad30.
# min-llvm-version: 12.0

all:
	$(RUSTC) -Z unstable-options -C split-debuginfo=packed -C debuginfo=2 foo.rs -g
	rm $(TMPDIR)/foo.dwp
	rm $(TMPDIR)/$(call BIN,foo)
