include ../tools.mk

# ignore-windows
# ignore-freebsd
# FIXME: on windows `rustc --dep-info` produces Makefile dependency with
# windows native paths (e.g. `c:\path\to\libfoo.a`)
# but msys make seems to fail to recognize such paths, so test fails.

all:
	cp *.rs $(TMPDIR)
	$(RUSTC) --emit dep-info,link --crate-type=lib $(TMPDIR)/lib.rs
	sleep 2
	touch $(TMPDIR)/foo.rs
	-rm -f $(TMPDIR)/done
	$(MAKE) -drf Makefile.foo
	sleep 2
	rm $(TMPDIR)/done
	pwd
	$(MAKE) -drf Makefile.foo
	rm $(TMPDIR)/done && exit 1 || exit 0

	# When a source file is deleted `make` should still work
	rm $(TMPDIR)/bar.rs
	cp $(TMPDIR)/lib2.rs $(TMPDIR)/lib.rs
	$(MAKE) -drf Makefile.foo
