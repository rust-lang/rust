include ../tools.mk

all:
	touch $(TMPDIR)/libfoo.a
	echo | $(RUSTC) - --crate-type=rlib -lstatic=foo 2>&1 | $(CGREP) "failed to add native library"
