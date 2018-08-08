-include ../../run-make-fulldeps/tools.mk

# How to run this
# $ ./x.py dist --target x86_64-unknown-linux-gnu,thumbv7m-none-eabi --exclude src/doc
# $ ./x.py test --target thumbv7m-none-eabi src/test/run-make

ifeq ($(TARGET),thumbv7m-none-eabi)

#TMP_DIR := $(shell mktemp -d)
TMP_DIR := /tmp/safe_place
CRATE := cortex-m
RUSTC := $(RUSTC_ORIGINAL)
LD_LIBRARY_PATH := $(HOST_RPATH_DIR)

all:
	env > env.txt
	mkdir -p $(TMP_DIR)
	-cd $(TMP_DIR) && rm -rf $(CRATE)
	cd $(TMP_DIR) && $(CARGO) clone $(CRATE) --vers 0.5.0
	cd $(TMP_DIR) && cd $(CRATE) && $(CARGO) build -j 1 --target $(TARGET) -v
else

all:

endif
