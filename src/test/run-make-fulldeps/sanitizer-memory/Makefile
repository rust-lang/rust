-include ../tools.mk

all:
ifeq ($(TARGET),x86_64-unknown-linux-gnu)
ifdef SANITIZER_SUPPORT
	$(RUSTC) -g -Z sanitizer=memory -Z print-link-args uninit.rs | $(CGREP) librustc_msan
	$(TMPDIR)/uninit 2>&1 | $(CGREP) use-of-uninitialized-value
endif
endif

