-include ../tools.mk

# needs-sanitizer-support
# only-linux
# only-x86_64
# ignore-test
# FIXME(#46126) ThinLTO for libstd broke this test

all:
	$(RUSTC) -C opt-level=1 -g -Z sanitizer=leak -Z print-link-args leak.rs | $(CGREP) librustc_lsan
	$(TMPDIR)/leak 2>&1 | $(CGREP) 'detected memory leaks'
