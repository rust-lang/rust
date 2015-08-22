-include ../tools.mk

TO_LINK := $(call DYLIB,bar)
LINK_ARG = $(TO_LINK:dll=lib)

all:
	$(RUSTC) foo.rs
	$(RUSTC) bar.rs
	$(CC) main.c $(call OUT_EXE,main) $(LINK_ARG) $(EXTRACFLAGS)
	rm $(TMPDIR)/*.rlib
	rm $(call DYLIB,foo)
	$(call RUN,main)
