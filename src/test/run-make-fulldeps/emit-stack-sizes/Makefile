-include ../tools.mk

ifdef IS_WINDOWS
# Do nothing on MSVC.
all:
	exit 0
else
# check that the .stack_sizes section is generated
all:
	$(RUSTC) -C opt-level=3 -Z emit-stack-sizes --emit=obj foo.rs
	size -A $(TMPDIR)/foo.o | $(CGREP) .stack_sizes
endif
