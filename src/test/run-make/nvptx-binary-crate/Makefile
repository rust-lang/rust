-include ../../run-make-fulldeps/tools.mk

ifeq ($(TARGET),nvptx64-nvidia-cuda)
all:
	$(RUSTC) main.rs --crate-type="bin" --target $(TARGET) -O -C link-arg=--arch=sm_60 -o $(TMPDIR)/main.link_arg.ptx
	$(RUSTC) main.rs --crate-type="bin" --target $(TARGET) -O -C target-cpu=sm_60 -o $(TMPDIR)/main.target_cpu.ptx

	FileCheck main.rs --input-file $(TMPDIR)/main.link_arg.ptx
	FileCheck main.rs --input-file $(TMPDIR)/main.target_cpu.ptx
else
all:
endif
