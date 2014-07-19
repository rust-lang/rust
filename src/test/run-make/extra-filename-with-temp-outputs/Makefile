-include ../tools.mk

all:
	$(RUSTC) -C extra-filename=bar foo.rs -C save-temps
	rm $(TMPDIR)/foobar.o
	rm $(TMPDIR)/$(call BIN,foobar)
