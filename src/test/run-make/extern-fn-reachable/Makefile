-include ../tools.mk

all:
	$(RUSTC) dylib.rs -o $(TMPDIR)/libdylib.so
	$(RUSTC) main.rs
	$(call RUN,main)
