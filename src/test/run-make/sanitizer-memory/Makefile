-include ../tools.mk

# NOTE the memory sanitizer only supports x86_64 linux
ifndef IS_WINDOWS
ifeq ($(shell uname),Linux)
ifeq ($(shell uname -m),x86_64)
all:
	$(RUSTC) -g -Z sanitizer=memory -Z print-link-args uninit.rs | grep -q librustc_msan
	$(TMPDIR)/uninit 2>&1 | grep -q use-of-uninitialized-value
else
all:

endif
else
all:

endif
else
all:

endif
