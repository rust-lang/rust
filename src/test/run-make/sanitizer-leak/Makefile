-include ../tools.mk

# NOTE the leak sanitizer only supports x86_64 linux
# Also, this particular sanitizer sometimes doesn't work so we are not going to
# run the binary
ifndef IS_WINDOWS
ifeq ($(shell uname),Linux)
ifeq ($(shell uname -m),x86_64)
all:
	$(RUSTC) -C opt-level=1 -g -Z sanitizer=leak -Z print-link-args leak.rs | grep -q librustc_lsan
	$(TMPDIR)/leak 2>&1 | grep -q 'detected memory leaks'
else
all:

endif
else
all:

endif
else
all:

endif
