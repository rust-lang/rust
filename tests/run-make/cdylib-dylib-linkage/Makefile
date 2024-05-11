# This test checks that cdylibs can link against dylibs as dependencies, after this restriction was disabled.
# See https://github.com/rust-lang/rust/commit/72aaa3a414d17aa0c4f19feafa5bab5f84b60e63

# ignore-cross-compile
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
CLIBS := $(TMPDIR)/foo.dll.lib $(TMPDIR)/bar.dll.lib $(LIBSTD)
$(call RUN_BINFILE,foo): $(call DYLIB,foo)
	$(CC) $(CFLAGS) foo.c $(CLIBS) $(call OUT_EXE,foo)
else
CLIBS := -lfoo -lbar -l$(STD) -L $(TMPDIR) -L $(TARGET_SYSROOT)
$(call RUN_BINFILE,foo): $(call DYLIB,foo)
	$(CC) $(CFLAGS) foo.c $(CLIBS) -o $(call RUN_BINFILE,foo)
endif

$(call DYLIB,foo):
	$(RUSTC) -C prefer-dynamic bar.rs
	$(RUSTC) foo.rs
