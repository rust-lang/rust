-include ../tools.mk

# FIXME: ignore freebsd/windows
# on windows `rustc --dep-info` produces Makefile dependency with
# windows native paths (e.g. `c:\path\to\libfoo.a`)
# but msys make seems to fail to recognize such paths, so test fails.
ifneq ($(shell uname),FreeBSD)
ifndef IS_WINDOWS
all:
	$(RUSTC) --emit dep-info main.rs
	grep "input.txt" $(TMPDIR)/main.d
	grep "input.bin" $(TMPDIR)/main.d
else
all:

endif

else
all:

endif
