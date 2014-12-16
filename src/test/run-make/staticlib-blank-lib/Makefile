-include ../tools.mk

all:
	ar crus $(TMPDIR)/libfoo.a foo.rs
	ar d $(TMPDIR)/libfoo.a foo.rs
	$(RUSTC) foo.rs
