-include ../../run-make-fulldeps/tools.mk

#only-x86_64-fortanix-unknown-sgx

OBJDUMP="${S}/build/x86_64-unknown-linux-gnu/llvm/build/bin/llvm-objdump"
FILECHECK="${S}/build/x86_64-unknown-linux-gnu/llvm/build/bin/FileCheck"

all:
	$(RUSTC) --target ${TARGET} enclave.rs

	#TODO: re-enable check when newly compiled libunwind is used
	#${OBJDUMP} --disassemble-symbols=unw_getcontext --demangle $(TMPDIR)/enclave > $(TMPDIR)/unw_getcontext.asm
	#${FILECHECK} --input-file $(TMPDIR)/unw_getcontext.asm unw_getcontext.checks

	#TODO: re-enable check when newly compiled libunwind is used
	${OBJDUMP} --disassemble-symbols="libunwind::Registers_x86_64::jumpto()" --demangle $(TMPDIR)/enclave > $(TMPDIR)/jumpto.asm
	${FILECHECK} --input-file $(TMPDIR)/jumpto.asm jumpto.checks
