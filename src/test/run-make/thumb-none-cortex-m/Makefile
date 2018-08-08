-include ../../run-make-fulldeps/tools.mk

# How to run this
# $ ./x.py test --target thumbv7m-none-eabi src/test/run-make

ifeq ($(TARGET),thumbv7m-none-eabi)

# We need to be outside of 'src' dir in order to run cargo
WORK_DIR := $(RUST_TEST_TMPDIR)/run-make/$(TARGET)

CRATE := cortex-m
CRATE_VER := 0.5.0

RUSTC := $(RUSTC_ORIGINAL)
LD_LIBRARY_PATH := $(HOST_RPATH_DIR)

all:
	env
	mkdir -p $(WORK_DIR)
	-cd $(WORK_DIR) && rm -rf $(CRATE)
	cd $(WORK_DIR) && $(CARGO) clone $(CRATE) --vers $(CRATE_VER)
	cd $(WORK_DIR) && cd $(CRATE) && $(CARGO) build -j 1 --target $(TARGET) -v
else

all:

endif
