-include ../../run-make-fulldeps/tools.mk

ifeq ($(TARGET),nvptx64-nvidia-cuda)
all:
	$(RUSTC) kernel.rs --crate-type="rlib" --emit asm,llvm-ir -O --target $(TARGET)
	FileCheck kernel.rs --input-file $(TMPDIR)/kernel.s
else
all:
endif
