-include ../../run-make-fulldeps/tools.mk

ifeq ($(TARGET),nvptx64-nvidia-cuda)
all:
	$(RUSTC) dep.rs --crate-type="rlib" --target $(TARGET)
	$(RUSTC) kernel.rs --crate-type="cdylib" -O --target $(TARGET)
	FileCheck kernel.rs --input-file $(TMPDIR)/kernel.ptx
else
all:
endif
