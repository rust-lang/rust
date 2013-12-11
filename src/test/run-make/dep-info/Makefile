-include ../tools.mk
all:
	$(RUSTC) --dep-info --lib lib.rs
	sleep 1
	touch foo.rs
	-rm -f $(TMPDIR)/done
	$(MAKE) -f Makefile.foo
	rm $(TMPDIR)/done
	pwd
	$(MAKE) -df Makefile.foo
	rm $(TMPDIR)/done && exit 1 || exit 0
