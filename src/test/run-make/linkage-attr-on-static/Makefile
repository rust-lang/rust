-include ../tools.mk

all:
	$(CC) foo.c -c -o $(TMPDIR)/foo.o
	$(AR) rcs $(TMPDIR)/libfoo.a $(TMPDIR)/foo.o
	$(RUSTC) bar.rs -lfoo -L $(TMPDIR)
	$(call RUN,bar) || exit 1

