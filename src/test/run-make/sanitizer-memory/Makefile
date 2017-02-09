-include ../tools.mk

ifdef SANITIZER_SUPPORT
all:
	$(RUSTC) -g -Z sanitizer=memory -Z print-link-args uninit.rs | grep -q librustc_msan
	$(TMPDIR)/uninit 2>&1 | grep -q use-of-uninitialized-value
else
all:

endif
