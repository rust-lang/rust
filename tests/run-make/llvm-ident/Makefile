include ../tools.mk

# only-linux

all:
	# `-Ccodegen-units=16 -Copt-level=2` is used here to trigger thin LTO
	# across codegen units to test deduplication of the named metadata
	# (see `LLVMRustPrepareThinLTOImport` for details).
	echo 'fn main(){}' | $(RUSTC) - --emit=link,obj -Csave-temps -Ccodegen-units=16 -Copt-level=2 --target=$(TARGET)

	# `llvm-dis` is used here since `--emit=llvm-ir` does not emit LLVM IR
	# for temporary outputs.
	"$(LLVM_BIN_DIR)"/llvm-dis $(TMPDIR)/*.bc

	# Check LLVM IR files (including temporary outputs) have `!llvm.ident`
	# named metadata, reusing the related codegen test.
	set -e; for f in $(TMPDIR)/*.ll; do \
		$(LLVM_FILECHECK) --input-file $$f ../../codegen/llvm-ident.rs; \
	done
