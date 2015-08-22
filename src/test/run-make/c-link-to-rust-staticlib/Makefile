-include ../tools.mk

# FIXME: ignore freebsd
ifneq ($(shell uname),FreeBSD)
all:
	$(RUSTC) foo.rs
	$(CC) bar.c $(TMPDIR)/libfoo.a $(call OUT_EXE,bar) \
		$(EXTRACFLAGS) $(EXTRACXXFLAGS)
	$(call RUN,bar)
	rm $(call STATICLIB,foo*)
	$(call RUN,bar)

else
all:

endif
