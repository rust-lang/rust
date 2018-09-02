-include ../../run-make-fulldeps/tools.mk

# How to run this
# $ ./x.py clean
# $ ./x.py test --target thumbv7m-none-eabi src/test/run-make

ifeq ($(TARGET),thumbv7m-none-eabi))

# For cargo setting
RUSTC := $(RUSTC_ORIGINAL)
LD_LIBRARY_PATH := $(HOST_RPATH_DIR)
# We need to be outside of 'src' dir in order to run cargo
WORK_DIR := $(TMPDIR)

HERE := $(shell pwd)

CRATE := lm3s6965evb
CRATE_URL := https://github.com/japaric/lm3s6965evb
CRATE_SHA1 := 9eeea58826438e84d89e9691a1bb0f85b03d377c
QEMU_CPU := cortex-m3
QEMU_MACHINE := lm3s6965evb

all:
	env
	mkdir -p $(WORK_DIR)
	-cd $(WORK_DIR) && rm -rf $(CRATE)
	cd $(WORK_DIR) && bash -x $(HERE)/../git_clone_sha1.sh $(CRATE) $(CRATE_URL) $(CRATE_SHA1)
	cd $(WORK_DIR)/$(CRATE) && $(CARGO) build --target $(TARGET) -v 
	cd $(WORK_DIR)/$(CRATE) && qemu-system-arm \
       	-cpu $(QEMU_CPU) \
       	-machine $(QEMU_MACHINE) \
       	-semihosting-config enable=on,target=native \
       	-nographic \
       	-kernel target/$(TARGET)/debug/$(CRATE) > out.txt
	cd $(WORK_DIR)/$(CRATE) && grep "x = 42" out.txt	
else

all:

endif
