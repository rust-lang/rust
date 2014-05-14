-include ../tools.mk

all:
	$(CC) -std=c99 test.c -c -o $(TMPDIR)/test.o
	$(AR) rcs $(TMPDIR)/libtest.a $(TMPDIR)/test.o
	$(RUSTC) testcrate.rs -L $(TMPDIR)
	$(RUSTC) test.rs -L $(TMPDIR)
	$(call RUN,test) || exit 1
