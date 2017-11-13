-include ../tools.mk

all:
ifeq ($(TARGET),x86_64-unknown-linux-gnu)
ifdef SANITIZER_SUPPORT
	$(RUSTC) -g -Z sanitizer=memory -Z print-link-args uninit.rs | tee $(TMPDIR)/out
	grep -q librustc_msan $(TMPDIR)/out
	$(TMPDIR)/uninit 2>&1 | tee $(TMPDIR)/out
	grep -q use-of-uninitialized-value $(TMPDIR)/out
endif
endif

