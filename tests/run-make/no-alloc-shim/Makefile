include ../tools.mk

# ignore-cross-compile
# ignore-msvc FIXME(bjorn3) can't figure out how to link with the MSVC toolchain

TARGET_LIBDIR = $$($(RUSTC) --print target-libdir)

all:
	$(RUSTC) foo.rs --crate-type bin --emit obj -Cpanic=abort
ifdef IS_MSVC
	$(CC) $(CFLAGS) $(TMPDIR)/foo.o $(call OUT_EXE,foo) /link $(TARGET_LIBDIR)/liballoc-*.rlib $(TARGET_LIBDIR)/libcore-*.rlib $(TARGET_LIBDIR)/libcompiler_builtins-*.rlib
	$(call OUT_EXE,foo)
else
	$(CC) $(CFLAGS) $(TMPDIR)/foo.o $(TARGET_LIBDIR)/liballoc-*.rlib $(TARGET_LIBDIR)/libcore-*.rlib $(TARGET_LIBDIR)/libcompiler_builtins-*.rlib -o $(call RUN_BINFILE,foo)
	$(call RUN_BINFILE,foo)
endif

	# Check that linking without __rust_no_alloc_shim_is_unstable defined fails
	$(RUSTC) foo.rs --crate-type bin --emit obj -Cpanic=abort --cfg check_feature_gate
ifdef IS_MSVC
	$(CC) $(CFLAGS) $(TMPDIR)/foo.o $(call OUT_EXE,foo) /link $(TARGET_LIBDIR)/liballoc-*.rlib $(TARGET_LIBDIR)/libcore-*.rlib $(TARGET_LIBDIR)/libcompiler_builtins-*.rlib || exit 0 && exit 1
else
	$(CC) $(CFLAGS) $(TMPDIR)/foo.o $(TARGET_LIBDIR)/liballoc-*.rlib $(TARGET_LIBDIR)/libcore-*.rlib $(TARGET_LIBDIR)/libcompiler_builtins-*.rlib -o $(call RUN_BINFILE,foo) || exit 0 && exit 1
endif
