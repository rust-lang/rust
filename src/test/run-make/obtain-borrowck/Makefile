-include ../../run-make-fulldeps/tools.mk

# This example shows how to implement a rustc driver that retrieves MIR bodies
# together with the borrow checker information.

# How to run this
# $ ./x.py test src/test/run-make/obtain-borrowck

DRIVER_BINARY := "$(TMPDIR)"/driver
BUILD_STAGE := $(RUST_BUILD_STAGE:$(TARGET)=)
RUSTC_SRC_LIB_DEPS := $(BUILD_DIR)/$(TARGET)/$(BUILD_STAGE)rustc/release/deps/
RUSTC_SRC_LIB := $(BUILD_DIR)/$(TARGET)/$(BUILD_STAGE)rustc/$(TARGET)/release/deps/
SYSROOT := $(shell $(RUSTC) --print sysroot)
DRIVER_RPATH_ENV = \
	$(LD_LIB_PATH_ENVVAR)="${RUSTC_SRC_LIB}:$(HOST_RPATH_DIR)/rustlib/$(TARGET)/lib/:$($(LD_LIB_PATH_ENVVAR))"

all:
	$(RUSTC) -L $(RUSTC_SRC_LIB) -L $(RUSTC_SRC_LIB_DEPS) driver.rs -o "$(DRIVER_BINARY)"
	$(DRIVER_RPATH_ENV) "$(DRIVER_BINARY)" --sysroot $(SYSROOT) test.rs -o "$(TMPDIR)/driver_test" > "$(TMPDIR)"/output.stdout

ifdef RUSTC_BLESS_TEST
	cp "$(TMPDIR)"/output.stdout output.stdout
else
	$(DIFF) output.stdout "$(TMPDIR)"/output.stdout
endif
