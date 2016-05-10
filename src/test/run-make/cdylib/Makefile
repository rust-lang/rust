include ../tools.mk

all:
	$(RUSTC) bar.rs -O
	$(RUSTC) foo.rs -O
	$(CC) $(CFLAGS) foo.c -lfoo -o $(call RUN_BINFILE,foo) -L $(TMPDIR)
	$(call RUN,foo)
	rm $(call DYLIB,foo)
	$(RUSTC) foo.rs -C lto
	$(call RUN,foo)
