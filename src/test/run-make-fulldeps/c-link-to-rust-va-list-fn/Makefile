include ../tools.mk

all:
	$(RUSTC) checkrust.rs
	$(CC) test.c $(call STATICLIB,checkrust) $(call OUT_EXE,test) $(EXTRACFLAGS)
	$(call RUN,test)
