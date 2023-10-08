# Test the behavior of #[link(.., kind = "raw-dylib")] with alternative calling conventions.

# only-x86
# only-windows

include ../tools.mk

all:
	$(RUSTC) --crate-type bin driver.rs -L "$(TMPDIR)"
	$(call COMPILE_OBJ,"$(TMPDIR)"/extern.obj,extern.c)
ifdef IS_MSVC
	$(CC) "$(TMPDIR)"/extern.obj extern.msvc.def -link -dll -out:"$(TMPDIR)"/extern.dll -noimplib
else
	$(CC) "$(TMPDIR)"/extern.obj extern.gnu.def --no-leading-underscore -shared -o "$(TMPDIR)"/extern.dll
endif
	"$(TMPDIR)"/driver > "$(TMPDIR)"/output.txt
	$(RUSTC_TEST_OP) "$(TMPDIR)"/output.txt output.txt
