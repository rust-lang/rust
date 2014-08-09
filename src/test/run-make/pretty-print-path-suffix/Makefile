-include ../tools.mk

all:
	$(RUSTC) -o $(TMPDIR)/foo.out --pretty normal=foo input.rs
	$(RUSTC) -o $(TMPDIR)/nest_foo.out --pretty normal=nest::foo input.rs
	$(RUSTC) -o $(TMPDIR)/foo_method.out --pretty normal=foo_method input.rs
	diff -u $(TMPDIR)/foo.out foo.pp
	diff -u $(TMPDIR)/nest_foo.out nest_foo.pp
	diff -u $(TMPDIR)/foo_method.out foo_method.pp
