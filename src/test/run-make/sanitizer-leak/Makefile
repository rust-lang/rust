-include ../tools.mk

all:
ifeq ($(TARGET),x86_64-unknown-linux-gnu)
ifdef SANITIZER_SUPPORT
	$(RUSTC) -C opt-level=1 -g -Z sanitizer=leak -Z print-link-args leak.rs | grep -q librustc_lsan
	$(TMPDIR)/leak 2>&1 | grep -q 'detected memory leaks'
endif
endif

