-include ../tools.mk

all:
	$(RUSTC) foo.rs
	touch $(call DYLIB,foo-something-special)
	touch $(call DYLIB,foo-something-special2)
	$(RUSTC) bar.rs
