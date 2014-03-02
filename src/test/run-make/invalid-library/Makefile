-include ../tools.mk

all:
	touch $(TMPDIR)/rust.metadata.bin
	ar crus $(TMPDIR)/libfoo-ffffffff-1.0.rlib $(TMPDIR)/rust.metadata.bin
	$(RUSTC) foo.rs 2>&1 | grep "can't find crate for"
