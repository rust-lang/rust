-include ../tools.mk

all:
	$(RUSTC) -o $(TMPDIR)/foo.out -Z unstable-options --unpretty hir=foo input.rs
	$(RUSTC) -o $(TMPDIR)/nest_foo.out -Z unstable-options --unpretty hir=nest::foo input.rs
	$(RUSTC) -o $(TMPDIR)/foo_method.out -Z unstable-options --unpretty hir=foo_method input.rs
	diff -u $(TMPDIR)/foo.out foo.pp
	diff -u $(TMPDIR)/nest_foo.out nest_foo.pp
	diff -u $(TMPDIR)/foo_method.out foo_method.pp
