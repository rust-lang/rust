-include ../tools.mk

all:
	$(CC) -c test.c -o $(call STATICLIB,test) $(EXTRACFLAGS) $(EXTRACXXFLAGS)
	$(RUSTC) nonclike.rs -L$(TMPDIR) -ltest
	$(call RUN,nonclike)
