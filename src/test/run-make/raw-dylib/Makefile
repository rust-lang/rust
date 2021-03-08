# Test the behavior of #[link(.., kind = "raw-dylib")] on windows-msvc

# only-windows
# only-msvc

-include ../../run-make-fulldeps/tools.mk

all:
	$(call COMPILE_OBJ,"$(TMPDIR)"/extern_1.obj,extern_1.c)
	$(call COMPILE_OBJ,"$(TMPDIR)"/extern_2.obj,extern_2.c)
	$(CC) "$(TMPDIR)"/extern_1.obj -link -dll -out:"$(TMPDIR)"/extern_1.dll
	$(CC) "$(TMPDIR)"/extern_2.obj -link -dll -out:"$(TMPDIR)"/extern_2.dll
	$(RUSTC) --crate-type lib --crate-name raw_dylib_test lib.rs
	$(RUSTC) --crate-type bin driver.rs -L "$(TMPDIR)"
	"$(TMPDIR)"/driver > "$(TMPDIR)"/output.txt

ifdef RUSTC_BLESS_TEST
	cp "$(TMPDIR)"/output.txt output.txt
else
	$(DIFF) output.txt "$(TMPDIR)"/output.txt
endif
