include ../tools.mk

all:
	$(RUSTC) --crate-type=staticlib nonclike.rs
	$(CC) test.c $(call STATICLIB,nonclike) $(call OUT_EXE,test) \
		$(EXTRACFLAGS) $(EXTRACXXFLAGS)
	$(call RUN,test)
