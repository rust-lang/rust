-include ../tools.mk

all:
	$(RUSTC) foo.rs -C metadata=a -C extra-filename=-a
	$(RUSTC) foo.rs -C metadata=b -C extra-filename=-b
	$(RUSTC) bar.rs \
		--extern foo1=$(TMPDIR)/libfoo-a.rlib \
		--extern foo2=$(TMPDIR)/libfoo-b.rlib \
		--print link-args
	$(call RUN,bar)
