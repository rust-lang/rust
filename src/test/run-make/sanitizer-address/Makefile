-include ../tools.mk

# NOTE the address sanitizer only supports x86_64 linux
ifndef IS_WINDOWS
ifeq ($(shell uname),Linux)
ifeq ($(shell uname -m),x86_64)
all:
	$(RUSTC) -g -Z sanitizer=address -Z print-link-args overflow.rs | grep -q librustc_asan
	$(TMPDIR)/overflow 2>&1 | grep -q stack-buffer-overflow
else
all:

endif
else
all:

endif
else
all:

endif
