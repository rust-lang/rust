-include ../tools.mk

# needs-sanitizer-support

all:
	$(RUSTC) -O -Z sanitizer=leak -Z print-link-args leak.rs | $(CGREP) rustc_rt.lsan
	$(TMPDIR)/leak 2>&1 | $(CGREP) 'detected memory leaks'
