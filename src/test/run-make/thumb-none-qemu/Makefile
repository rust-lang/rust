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

all:
	bash script.sh
else
all:
endif
