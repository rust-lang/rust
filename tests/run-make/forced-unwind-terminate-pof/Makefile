# ignore-cross-compile
# only-linux
include ../tools.mk

all: foo
	$(call RUN,foo) | $(CGREP) -v "cannot unwind"

foo: foo.rs
	$(RUSTC) $<
