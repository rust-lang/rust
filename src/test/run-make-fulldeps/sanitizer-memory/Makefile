-include ../tools.mk

# only-linux
# only-x86_64

all:
ifdef SANITIZER_SUPPORT
	$(RUSTC) -g -Z sanitizer=memory -Z print-link-args uninit.rs | $(CGREP) librustc_msan
	$(TMPDIR)/uninit 2>&1 | $(CGREP) use-of-uninitialized-value
endif
