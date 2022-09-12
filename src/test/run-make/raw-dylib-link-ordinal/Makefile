# Test the behavior of #[link(.., kind = "raw-dylib")] and #[link_ordinal] on windows-msvc

# only-windows

include ../../run-make-fulldeps/tools.mk

all:
	$(RUSTC) --crate-type lib --crate-name raw_dylib_test lib.rs
	$(RUSTC) --crate-type bin driver.rs -L "$(TMPDIR)"
	$(call COMPILE_OBJ,"$(TMPDIR)"/exporter.obj,exporter.c)
ifdef IS_MSVC
	$(CC) "$(TMPDIR)"/exporter.obj exporter.def -link -dll -out:"$(TMPDIR)"/exporter.dll -noimplib
else
	$(CC) "$(TMPDIR)"/exporter.obj exporter.def -shared -o "$(TMPDIR)"/exporter.dll
endif
	"$(TMPDIR)"/driver > "$(TMPDIR)"/output.txt

ifdef RUSTC_BLESS_TEST
	cp "$(TMPDIR)"/output.txt output.txt
else
	$(DIFF) output.txt "$(TMPDIR)"/output.txt
endif
