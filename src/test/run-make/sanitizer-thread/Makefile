-include ../tools.mk

# NOTE the leak sanitizer only supports x86_64 linux
ifndef IS_WINDOWS
ifeq ($(shell uname),Linux)
ifeq ($(shell uname -m),x86_64)
all:
	$(RUSTC) -g -Z sanitizer=thread -Z print-link-args racy.rs | grep -q librustc_tsan
	$(TMPDIR)/racy 2>&1 | grep -q 'data race'
else
all:

endif
else
all:

endif
else
all:

endif
