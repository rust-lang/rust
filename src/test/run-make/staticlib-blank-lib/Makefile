-include ../tools.mk

all:
	ar crus libfoo.a foo.rs
	ar d libfoo.a foo.rs
	$(RUSTC) foo.rs
