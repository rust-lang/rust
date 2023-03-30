include ../tools.mk

TO_LINK := $(call DYLIB,bar)
ifdef IS_MSVC
LINK_ARG = $(TO_LINK:dll=dll.lib)
else
LINK_ARG = $(TO_LINK)
endif

all:
	$(RUSTC) foo.rs
	$(RUSTC) bar.rs
	$(CC) main.c $(call OUT_EXE,main) $(LINK_ARG) $(EXTRACFLAGS)
	rm $(TMPDIR)/*.rlib
	rm $(call DYLIB,foo)
	$(call RUN,main)
