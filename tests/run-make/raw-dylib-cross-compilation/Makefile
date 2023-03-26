# Tests that raw-dylib cross compilation works correctly

# only-gnu
# needs-i686-dlltool
# needs-x86_64-dlltool

# i686 dlltool.exe can't product x64 binaries.
# ignore-i686-pc-windows-gnu

include ../../run-make-fulldeps/tools.mk

all:
	# Build as x86 and make sure that we have x86 objects only.
	$(RUSTC) --crate-type lib --crate-name i686_raw_dylib_test --target i686-pc-windows-gnu lib.rs
	"$(LLVM_BIN_DIR)"/llvm-objdump -a $(TMPDIR)/libi686_raw_dylib_test.rlib > $(TMPDIR)/i686.objdump.txt
	$(CGREP) "file format coff-i386" < $(TMPDIR)/i686.objdump.txt
	$(CGREP) -v "file format coff-x86-64" < $(TMPDIR)/i686.objdump.txt
	# Build as x64 and make sure that we have x64 objects only.
	$(RUSTC) --crate-type lib --crate-name x64_raw_dylib_test --target x86_64-pc-windows-gnu lib.rs
	"$(LLVM_BIN_DIR)"/llvm-objdump -a $(TMPDIR)/libx64_raw_dylib_test.rlib > $(TMPDIR)/x64.objdump.txt
	$(CGREP) "file format coff-x86-64" < $(TMPDIR)/x64.objdump.txt
	$(CGREP) -v "file format coff-i386" < $(TMPDIR)/x64.objdump.txt
