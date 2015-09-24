-include ../tools.mk

# FIXME: ignore freebsd
ifneq ($(shell uname),FreeBSD)
all:
	$(RUSTC) foo.rs
	cp $(TMPDIR)/libfoo.a $(call NATIVE_STATICLIB,foo2)
	$(CC) bar.c $(call NATIVE_STATICLIB,foo2) $(call OUT_EXE,bar) \
		$(EXTRACFLAGS) $(EXTRACXXFLAGS)
	$(call RUN,bar)
	rm $(call STATICLIB,foo*)
	$(call RUN,bar)

else
all:

endif
