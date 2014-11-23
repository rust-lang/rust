-include ../tools.mk

all:
	$(RUSTC) foo.rs
	$(RUSTC) bar.rs
	$(CC) main.c -o $(call RUN_BINFILE,main) -lbar $(EXTRACFLAGS)
	rm $(TMPDIR)/*.rlib
	rm $(call DYLIB,foo)
	$(call RUN,main)
