# Regression test for calling an inline function that uses a raw-dylib function.

# only-windows

include ../../run-make-fulldeps/tools.mk

all:
	$(RUSTC) --crate-type dylib --crate-name raw_dylib_test lib.rs -C prefer-dynamic
	$(RUSTC) --crate-type dylib --crate-name raw_dylib_test_wrapper lib_wrapper.rs -C prefer-dynamic
	$(RUSTC) --crate-type bin driver.rs -L "$(TMPDIR)" -C prefer-dynamic
	# Make sure we don't find an import to the functions we expect to be inlined.
	"$(LLVM_BIN_DIR)"/llvm-objdump -p $(TMPDIR)/driver.exe | $(CGREP) -v -e "inline_library_function"
	"$(LLVM_BIN_DIR)"/llvm-objdump -p $(TMPDIR)/driver.exe | $(CGREP) -v -e "inline_library_function_calls_inline"
	# Make sure we do find an import to the functions we expect to be imported.
	"$(LLVM_BIN_DIR)"/llvm-objdump -p $(TMPDIR)/driver.exe | $(CGREP) -e "library_function"
	$(call COMPILE_OBJ,"$(TMPDIR)"/extern_1.obj,extern_1.c)
	$(call COMPILE_OBJ,"$(TMPDIR)"/extern_2.obj,extern_2.c)
ifdef IS_MSVC
	$(CC) "$(TMPDIR)"/extern_1.obj -link -dll -out:"$(TMPDIR)"/extern_1.dll -noimplib
	$(CC) "$(TMPDIR)"/extern_2.obj -link -dll -out:"$(TMPDIR)"/extern_2.dll -noimplib
else
	$(CC) "$(TMPDIR)"/extern_1.obj -shared -o "$(TMPDIR)"/extern_1.dll
	$(CC) "$(TMPDIR)"/extern_2.obj -shared -o "$(TMPDIR)"/extern_2.dll
endif
	$(call RUN,driver) > "$(TMPDIR)"/output.txt

ifdef RUSTC_BLESS_TEST
	cp "$(TMPDIR)"/output.txt output.txt
else
	$(DIFF) output.txt "$(TMPDIR)"/output.txt
endif
