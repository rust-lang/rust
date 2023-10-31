include ../tools.mk

all:
	$(RUSTC) --emit=dep-info -Zunpretty=expanded with-dep.rs
	$(CGREP) "with-dep.rs" < $(TMPDIR)/with-dep.d
	-rm $(TMPDIR)/with-dep.d

	$(RUSTC) --emit=dep-info -Zunpretty=normal with-dep.rs
	! test -f $(TMPDIR)/with-dep.d
