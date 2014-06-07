-include ../tools.mk

# Apparently older versions of GCC segfault if -g is passed...
CC := $(CC:-g=)

all:
	$(RUSTC) foo.rs -Z lto
	ln -s $(call STATICLIB,foo-*) $(call STATICLIB,foo)
	$(CC) bar.c -lfoo -o $(call RUN_BINFILE,bar) $(EXTRACFLAGS) -lstdc++
	$(call RUN,bar)
