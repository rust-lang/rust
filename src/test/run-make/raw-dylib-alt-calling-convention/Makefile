# Test the behavior of #[link(.., kind = "raw-dylib")] with alternative calling conventions.

# only-i686-pc-windows-msvc

-include ../../run-make-fulldeps/tools.mk

all:
	$(call COMPILE_OBJ,"$(TMPDIR)"/extern.obj,extern.c)
	$(CC) "$(TMPDIR)"/extern.obj -link -dll -out:"$(TMPDIR)"/extern.dll
	$(RUSTC) --crate-type lib --crate-name raw_dylib_alt_calling_convention_test lib.rs
	$(RUSTC) --crate-type bin driver.rs -L "$(TMPDIR)"
	"$(TMPDIR)"/driver > "$(TMPDIR)"/output.txt

ifdef RUSTC_BLESS_TEST
	cp "$(TMPDIR)"/output.txt output.txt
else
	$(DIFF) output.txt "$(TMPDIR)"/output.txt
endif
