include ../tools.mk

all:
	touch $(TMPDIR)/lib.rmeta
	$(AR) crus $(TMPDIR)/libfoo-ffffffff-1.0.rlib $(TMPDIR)/lib.rmeta
	$(RUSTC) foo.rs 2>&1 | $(CGREP) "found invalid metadata"
