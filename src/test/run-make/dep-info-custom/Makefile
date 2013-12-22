-include ../tools.mk

all:
	$(RUSTC) --dep-info $(TMPDIR)/custom-deps-file.d --lib lib.rs
	sleep 1
	touch foo.rs
	-rm -f $(TMPDIR)/done
	$(MAKE) -drf Makefile.foo
	rm $(TMPDIR)/done
	pwd
	$(MAKE) -drf Makefile.foo
	rm $(TMPDIR)/done && exit 1 || exit 0
