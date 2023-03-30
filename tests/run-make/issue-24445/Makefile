include ../tools.mk

# only-linux

all:
	$(RUSTC) foo.rs
	$(CC) foo.c -lfoo -L $(TMPDIR) -Wl,--gc-sections -lpthread -ldl -o $(TMPDIR)/foo
	$(call RUN,foo)
	$(CC) foo.c -lfoo -L $(TMPDIR) -Wl,--gc-sections -lpthread -ldl -pie -fPIC -o $(TMPDIR)/foo
	$(call RUN,foo)
