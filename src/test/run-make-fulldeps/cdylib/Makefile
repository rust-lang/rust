include ../tools.mk

all: $(call RUN_BINFILE,foo)
	$(call RUN,foo)
	rm $(call DYLIB,foo)
	$(RUSTC) foo.rs -C lto
	$(call RUN,foo)

ifdef IS_MSVC
$(call RUN_BINFILE,foo): $(call DYLIB,foo)
	$(CC) $(CFLAGS) foo.c $(TMPDIR)/foo.dll.lib $(call OUT_EXE,foo)
else
$(call RUN_BINFILE,foo): $(call DYLIB,foo)
	$(CC) $(CFLAGS) foo.c -lfoo -o $(call RUN_BINFILE,foo) -L $(TMPDIR)
endif

$(call DYLIB,foo):
	$(RUSTC) bar.rs
	$(RUSTC) foo.rs
