-include ../tools.mk

EXTRAFLAGS := $(EXTRACFLAGS)

# FIXME: ignore freebsd
ifneq ($(shell uname),FreeBSD)
all:
	$(RUSTC) foo.rs
	$(CC) bar.c -lfoo -o $(call RUN_BINFILE,bar) $(EXTRAFLAGS) $(EXTRACXXFLAGS)
	$(call RUN,bar)
	rm $(call STATICLIB,foo*)
	$(call RUN,bar)

else
all:

endif
