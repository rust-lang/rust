-include ../tools.mk

# ignore-windows
# ignore-freebsd
# FIXME: on windows `rustc --dep-info` produces Makefile dependency with
# windows native paths (e.g. `c:\path\to\libfoo.a`)
# but msys make seems to fail to recognize such paths, so test fails.

all:
	$(RUSTC) --emit dep-info main.rs
	$(CGREP) "input.txt" "input.bin" "input.md" < $(TMPDIR)/main.d
