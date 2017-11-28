-include ../tools.mk

# FIXME(#46126) ThinLTO for libstd broke this test
ifeq (1,0)
all:
ifeq ($(TARGET),x86_64-unknown-linux-gnu)
ifdef SANITIZER_SUPPORT
	$(RUSTC) -C opt-level=1 -g -Z sanitizer=leak -Z print-link-args leak.rs | $(CGREP) librustc_lsan
	$(TMPDIR)/leak 2>&1 | $(CGREP) 'detected memory leaks'
endif
endif

else
all:
endif

