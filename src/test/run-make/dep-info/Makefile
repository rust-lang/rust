-include ../tools.mk

ifneq ($(shell uname),FreeBSD)
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
