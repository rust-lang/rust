include ../tools.mk

TARGET_SYSROOT := $(shell $(RUSTC) --print sysroot)/lib/rustlib/$(TARGET)/lib

ifdef IS_MSVC
LIBSTD := $(wildcard $(TARGET_SYSROOT)/libstd-*.dll.lib)
else
LIBSTD := $(wildcard $(TARGET_SYSROOT)/$(call DYLIB_GLOB,std))
STD := $(basename $(patsubst lib%,%, $(notdir $(LIBSTD))))
endif

all: $(call RUN_BINFILE,foo)
	$(call RUN,foo)

ifdef IS_MSVC
CLIBS := $(TMPDIR)/foo.lib $(TMPDIR)/bar.dll.lib $(LIBSTD)
$(call RUN_BINFILE,foo): $(call STATICLIB,foo)
	$(CC) $(CFLAGS) foo.c $(CLIBS) $(call OUT_EXE,foo)
else
CLIBS := $(TMPDIR)/libfoo.a -lbar -l$(STD) -L $(TMPDIR) -L $(TARGET_SYSROOT)
$(call RUN_BINFILE,foo): $(call STATICLIB,foo)
	$(CC) $(CFLAGS) foo.c $(CLIBS) -o $(call RUN_BINFILE,foo)
endif

$(call STATICLIB,foo):
	$(RUSTC) -C prefer-dynamic bar.rs
	$(RUSTC) foo.rs
