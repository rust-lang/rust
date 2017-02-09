-include ../tools.mk

ifdef SANITIZER_SUPPORT
all:
	$(RUSTC) -C opt-level=1 -g -Z sanitizer=leak -Z print-link-args leak.rs | grep -q librustc_lsan
	$(TMPDIR)/leak 2>&1 | grep -q 'detected memory leaks'
else
all:

endif
