-include ../tools.mk

# NOTE the address sanitizer only supports x86_64 linux
ifdef SANITIZER_SUPPORT
all:
	$(RUSTC) -g -Z sanitizer=address -Z print-link-args overflow.rs | grep -q librustc_asan
	$(TMPDIR)/overflow 2>&1 | grep -q stack-buffer-overflow
else
all:

endif
