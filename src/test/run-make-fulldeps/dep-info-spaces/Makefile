-include ../tools.mk

# ignore-windows
# ignore-freebsd
# FIXME: (windows: see `../dep-info/Makefile`)

all:
	cp lib.rs $(TMPDIR)/
	cp 'foo foo.rs' $(TMPDIR)/
	cp bar.rs $(TMPDIR)/
	$(RUSTC) --emit link,dep-info --crate-type=lib $(TMPDIR)/lib.rs
	sleep 1
	touch $(TMPDIR)/'foo foo.rs'
	-rm -f $(TMPDIR)/done
	$(MAKE) -drf Makefile.foo
	rm $(TMPDIR)/done
	pwd
	$(MAKE) -drf Makefile.foo
	rm $(TMPDIR)/done && exit 1 || exit 0
