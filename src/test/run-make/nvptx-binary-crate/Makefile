-include ../../run-make-fulldeps/tools.mk

ifeq ($(TARGET),nvptx64-nvidia-cuda)
all:
	$(RUSTC) main.rs -Clink-arg=--arch=sm_60 --crate-type="bin" -O --target $(TARGET)
	FileCheck main.rs --input-file $(TMPDIR)/main.ptx
else
all:
endif
