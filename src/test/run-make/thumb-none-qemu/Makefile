-include ../../run-make-fulldeps/tools.mk

# How to run this
# $ ./x.py clean
# $ ./x.py test --target thumbv7m-none-eabi src/test/run-make

ifneq (,$(filter $(TARGET),thumbv6m-none-eabi thumbv7m-none-eabi))

# For cargo setting
export RUSTC := $(RUSTC_ORIGINAL)
LD_LIBRARY_PATH := $(HOST_RPATH_DIR)
# We need to be outside of 'src' dir in order to run cargo
export WORK_DIR := $(TMPDIR)
export HERE := $(shell pwd)


# hint: we could set variables per $(TARGET) basis in order to support other targets.
export CRATE := cortex-m-rt
export CRATE_URL := https://github.com/rust-embedded/cortex-m-rt
export CRATE_SHA1 := 62972c8a89ff54b76f9ef0d600c1fcf7a233aabd

all:
	bash script.sh
else
all:
endif
