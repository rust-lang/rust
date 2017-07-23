-include ../tools.mk

all:
	$(RUSTC) -C extra-filename=bar foo.rs -C save-temps
	rm $(TMPDIR)/foobar.foo0.rust-cgu.o
	rm $(TMPDIR)/$(call BIN,foobar)
