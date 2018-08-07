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
HOST := x86_64-unknown-linux-gnu

all:
	# Workaround for error[E0463]: can't find crate for `core`
	-ln -sv $(SRC)/build/tmp/dist/rust-std-*-$(TARGET)/rust-std-$(TARGET)/lib/rustlib/$(TARGET)/lib/*.rlib $(SRC)/build/x86_64-unknown-linux-gnu/stage2/lib/rustlib/$(TARGET)/lib/
	# Workaround for error[E0463]: can't find crate for `std`
	-ln -sv $(SRC)/build/tmp/dist/rust-std-*-$(HOST)/rust-std-$(HOST)/lib/rustlib/$(HOST)/lib/*.rlib $(SRC)/build/x86_64-unknown-linux-gnu/stage2/lib/rustlib/$(HOST)/lib/
	env > env.txt
	mkdir -p $(TMP_DIR)
	-cd $(TMP_DIR) && rm -rf $(CRATE)
	cd $(TMP_DIR) && $(CARGO) clone $(CRATE) --vers 0.5.0
	cd $(TMP_DIR) && cd $(CRATE) && $(CARGO) build -j 1 --target $(TARGET) -v
else

all:

endif
