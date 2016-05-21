-include ../tools.mk

all:
	$(RUSTC) foo1.rs
	$(RUSTC) foo2.rs
	mkdir $(TMPDIR)/foo
	cp $(TMPDIR)/libfoo1.rlib $(TMPDIR)/foo/libfoo1.rlib
	$(RUSTC) bar.rs \
		--extern foo1=$(TMPDIR)/foo/libfoo1.rlib \
		--extern foo2=$(TMPDIR)/libfoo2.rlib
