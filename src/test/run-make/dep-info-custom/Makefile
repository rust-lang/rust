-include ../tools.mk

# FIXME: ignore freebsd/windows
# (windows: see `../dep-info/Makefile`)
ifneq ($(shell uname),FreeBSD)
ifndef IS_WINDOWS
all:
	$(RUSTC) --dep-info $(TMPDIR)/custom-deps-file.d --crate-type=lib lib.rs
	sleep 1
	touch foo.rs
	-rm -f $(TMPDIR)/done
	$(MAKE) -drf Makefile.foo
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
