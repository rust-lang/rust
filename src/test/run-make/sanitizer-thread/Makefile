-include ../tools.mk

ifdef SANITIZER_SUPPORT
all:
	$(RUSTC) -g -Z sanitizer=thread -Z print-link-args racy.rs | grep -q librustc_tsan
	$(TMPDIR)/racy 2>&1 | grep -q 'data race'
else
all:

endif
