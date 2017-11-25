-include ../tools.mk

LOG := $(TMPDIR)/log.txt

# FIXME(#46126) ThinLTO for libstd broke this test
ifeq (1,0)
all:
ifeq ($(TARGET),x86_64-unknown-linux-gnu)
ifdef SANITIZER_SUPPORT
	$(RUSTC) -C opt-level=1 -g -Z sanitizer=leak -Z print-link-args leak.rs | grep -q librustc_lsan
	$(TMPDIR)/leak 2>&1 | tee $(LOG)
	grep -q 'detected memory leaks' $(LOG)
endif
endif

else
all:
endif

