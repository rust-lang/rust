# Test the behavior of #[link(.., kind = "raw-dylib")], #[link_ordinal], and alternative calling conventions on i686 windows.

# only-x86
# only-windows

include ../tools.mk

all:
	$(RUSTC) --crate-type lib --crate-name raw_dylib_test lib.rs
	$(RUSTC) --crate-type bin driver.rs -L "$(TMPDIR)"
	$(call COMPILE_OBJ,"$(TMPDIR)"/exporter.obj,exporter.c)
ifdef IS_MSVC
	$(CC) "$(TMPDIR)"/exporter.obj exporter-msvc.def -link -dll -out:"$(TMPDIR)"/exporter.dll -noimplib
else
	$(CC) "$(TMPDIR)"/exporter.obj exporter-gnu.def -shared -o "$(TMPDIR)"/exporter.dll
endif
	"$(TMPDIR)"/driver > "$(TMPDIR)"/actual_output.txt
	$(RUSTC_TEST_OP) "$(TMPDIR)"/actual_output.txt expected_output.txt
