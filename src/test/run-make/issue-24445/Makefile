-include ../tools.mk

ifeq ($(UNAME),Linux)
all:
	$(RUSTC) foo.rs
	$(CC) foo.c -lfoo -L $(TMPDIR) -Wl,--gc-sections -lpthread -o $(TMPDIR)/foo
	$(call RUN,foo)
	$(CC) foo.c -lfoo -L $(TMPDIR) -Wl,--gc-sections -lpthread -pie -fPIC -o $(TMPDIR)/foo
	$(call RUN,foo)
else
all:
endif
