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


# hint: we could set variables per $(TARGET) basis in order to support other targets.
CRATE := cortex-m-rt
CRATE_URL := https://github.com/rust-embedded/cortex-m-rt
CRATE_SHA1 := 62972c8a89ff54b76f9ef0d600c1fcf7a233aabd

all:
	env
	mkdir -p $(WORK_DIR)
	-cd $(WORK_DIR) && rm -rf $(CRATE)
	cd $(WORK_DIR) && bash -x $(HERE)/../git_clone_sha1.sh $(CRATE) $(CRATE_URL) $(CRATE_SHA1)
	cd $(WORK_DIR)/$(CRATE) && $(CARGO) run --target $(TARGET) --example qemu           | grep "x = 42"
	cd $(WORK_DIR)/$(CRATE) && $(CARGO) run --target $(TARGET) --example qemu --release | grep "x = 42"
else

all:

endif
