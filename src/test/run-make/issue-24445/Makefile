-include ../tools.mk

ifeq ($(UNAME),Linux)
all:
	$(RUSTC) foo.rs
	$(CC) foo.c -lfoo -L $(TMPDIR) -Wl,--gc-sections -lpthread -ldl -o $(TMPDIR)/foo
	$(call RUN,foo)
	$(CC) foo.c -lfoo -L $(TMPDIR) -Wl,--gc-sections -lpthread -ldl -pie -fPIC -o $(TMPDIR)/foo
	$(call RUN,foo)
else
all:
endif
