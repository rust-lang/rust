-include ../tools.mk

# This feature only works when the output object format is ELF so we ignore
# macOS and Windows
ifdef IS_WINDOWS
# Do nothing on Windows.
all:
	exit 0
else ifneq (,$(filter $(TARGET),i686-apple-darwin x86_64-apple-darwin))
# Do nothing on macOS.
all:
	exit 0
else
# check that the .stack_sizes section is generated
# this test requires LLVM >= 6.0.0
vers = $(shell $(RUSTC) -Vv)
ifneq (,$(findstring LLVM version: 3,$(vers)))
all:
	exit 0
else ifneq (,$(findstring LLVM version: 4,$(vers)))
all:
	exit 0
else ifneq (,$(findstring LLVM version: 5,$(vers)))
all:
	exit 0
else
all:
	$(RUSTC) -C opt-level=3 -Z emit-stack-sizes --emit=obj foo.rs
	size -A $(TMPDIR)/foo.o | $(CGREP) .stack_sizes
endif
endif
