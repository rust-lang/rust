-include ../tools.mk

ifdef IS_WINDOWS
# Do nothing on MSVC.
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
