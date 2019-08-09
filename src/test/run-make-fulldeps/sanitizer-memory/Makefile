-include ../tools.mk

# needs-sanitizer-support
# only-linux
# only-x86_64

all:
	$(RUSTC) -g -Z sanitizer=memory -Z print-link-args uninit.rs | $(CGREP) librustc_msan
	$(TMPDIR)/uninit 2>&1 | $(CGREP) use-of-uninitialized-value
