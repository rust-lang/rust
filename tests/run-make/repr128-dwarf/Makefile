# ignore-windows
# This test should be replaced with one in tests/debuginfo once GDB or LLDB support 128-bit
# enums.

include ../tools.mk

all:
	$(RUSTC) -Cdebuginfo=2 lib.rs -o $(TMPDIR)/repr128.rlib
	"$(LLVM_BIN_DIR)"/llvm-dwarfdump -n U128A $(TMPDIR)/repr128.rlib | $(CGREP) "DW_AT_const_value	(<0x10> 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 )"
	"$(LLVM_BIN_DIR)"/llvm-dwarfdump -n U128B $(TMPDIR)/repr128.rlib | $(CGREP) "DW_AT_const_value	(<0x10> 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 )"
	"$(LLVM_BIN_DIR)"/llvm-dwarfdump -n U128C $(TMPDIR)/repr128.rlib | $(CGREP) "DW_AT_const_value	(<0x10> 00 00 00 00 00 00 00 00 01 00 00 00 00 00 00 00 )"
	"$(LLVM_BIN_DIR)"/llvm-dwarfdump -n U128D $(TMPDIR)/repr128.rlib | $(CGREP) "DW_AT_const_value	(<0x10> ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff )"
	"$(LLVM_BIN_DIR)"/llvm-dwarfdump -n I128A $(TMPDIR)/repr128.rlib | $(CGREP) "DW_AT_const_value	(<0x10> 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 )"
	"$(LLVM_BIN_DIR)"/llvm-dwarfdump -n I128B $(TMPDIR)/repr128.rlib | $(CGREP) "DW_AT_const_value	(<0x10> ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff )"
	"$(LLVM_BIN_DIR)"/llvm-dwarfdump -n I128C $(TMPDIR)/repr128.rlib | $(CGREP) "DW_AT_const_value	(<0x10> 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 80 )"
	"$(LLVM_BIN_DIR)"/llvm-dwarfdump -n I128D $(TMPDIR)/repr128.rlib | $(CGREP) "DW_AT_const_value	(<0x10> ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff 7f )"
