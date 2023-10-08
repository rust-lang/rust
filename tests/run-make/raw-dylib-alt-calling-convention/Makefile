# Test the behavior of #[link(.., kind = "raw-dylib")] with alternative calling conventions.

# only-x86
# only-windows

include ../tools.mk

all:
	$(RUSTC) --crate-type lib --crate-name raw_dylib_alt_calling_convention_test lib.rs
	$(RUSTC) --crate-type bin driver.rs -L "$(TMPDIR)"
	$(call COMPILE_OBJ,"$(TMPDIR)"/extern.obj,extern.c)
ifdef IS_MSVC
	$(CC) "$(TMPDIR)"/extern.obj -link -dll -out:"$(TMPDIR)"/extern.dll -noimplib
else
	$(CC) "$(TMPDIR)"/extern.obj -shared -o "$(TMPDIR)"/extern.dll
endif

	"$(TMPDIR)"/driver > "$(TMPDIR)"/output.txt
	$(RUSTC_TEST_OP) "$(TMPDIR)"/output.txt output.txt

ifdef IS_MSVC
	"$(TMPDIR)"/driver true > "$(TMPDIR)"/output.msvc.txt
	$(RUSTC_TEST_OP) "$(TMPDIR)"/output.msvc.txt output.msvc.txt
endif
