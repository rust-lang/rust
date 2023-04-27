include ../tools.mk

# ignore-cross-compile
# ignore-msvc FIXME(bjorn3) can't figure out how to link with the MSVC toolchain
# ignore-wasm wasm doesn't support dynamic libraries

all:
	$(RUSTC) -C prefer-dynamic bar.rs
	$(RUSTC) foo.rs --crate-type staticlib --print native-static-libs \
		-Z staticlib-allow-rdylib-deps 2>&1 | grep 'note: native-static-libs: ' \
		| sed 's/note: native-static-libs: \(.*\)/\1/' > $(TMPDIR)/libs.txt
	cat $(TMPDIR)/libs.txt

ifdef IS_MSVC
	$(CC) $(CFLAGS) /c foo.c /Fo:$(TMPDIR)/foo.o
	$(RUSTC_LINKER) $(TMPDIR)/foo.o $(TMPDIR)/foo.lib $$(cat $(TMPDIR)/libs.txt) $(call OUT_EXE,foo)
else
	$(CC) $(CFLAGS) foo.c -L $(TMPDIR) -lfoo $$(cat $(TMPDIR)/libs.txt) -o $(call RUN_BINFILE,foo)
endif

	$(call RUN,foo)
