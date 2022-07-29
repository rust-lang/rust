# Test the behavior of #[link(.., kind = "raw-dylib")], #[link_ordinal], and alternative calling conventions on i686 windows.

# only-x86
# only-windows

-include ../../run-make-fulldeps/tools.mk

all:
	$(call COMPILE_OBJ,"$(TMPDIR)"/exporter.obj,exporter.c)
ifdef IS_MSVC
	$(CC) "$(TMPDIR)"/exporter.obj exporter-msvc.def -link -dll -out:"$(TMPDIR)"/exporter.dll
else
	$(CC) "$(TMPDIR)"/exporter.obj exporter-gnu.def -shared -o "$(TMPDIR)"/exporter.dll
endif
	$(RUSTC) --crate-type lib --crate-name raw_dylib_test lib.rs
	$(RUSTC) --crate-type bin driver.rs -L "$(TMPDIR)"
	"$(TMPDIR)"/driver > "$(TMPDIR)"/actual_output.txt

ifdef RUSTC_BLESS_TEST
	cp "$(TMPDIR)"/actual_output.txt expected_output.txt
else
	$(DIFF) expected_output.txt "$(TMPDIR)"/actual_output.txt
endif
