-include ../tools.mk

# FIXME: ignore freebsd/windows
# on windows `rustc --dep-info` produces Makefile dependency with
# windows native paths (e.g. `c:\path\to\libfoo.a`)
# but msys make seems to fail to recognize such paths, so test fails.
ifneq ($(shell uname),FreeBSD)
ifndef IS_WINDOWS
all:
	$(RUSTC) --dep-info --crate-type=lib lib.rs
	sleep 2
	touch foo.rs
	-rm -f $(TMPDIR)/done
	$(MAKE) -drf Makefile.foo
	sleep 2
	rm $(TMPDIR)/done
	pwd
	$(MAKE) -drf Makefile.foo
	rm $(TMPDIR)/done && exit 1 || exit 0
else
all:

endif

else
all:

endif
