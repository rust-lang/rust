# Don't use the definitions
#-include ../../run-make-fulldeps/tools.mk

ifeq ($(TARGET),thumbv7m-none-eabi)

#TMP_DIR := $(shell mktemp -d)
TMP_DIR := /tmp/safe_place
CRATE := cortex-m
LD_LIBRARY_PATH=/home/sekineh/rustme9/build/x86_64-unknown-linux-gnu/stage2/lib/rustlib/thumbv7m-none-eabi/lib

all:
	env > env.txt
	mkdir -p $(TMP_DIR)
	-cd $(TMP_DIR) && rm -rf $(CRATE)
	cd $(TMP_DIR) && $(CARGO) clone $(CRATE) --vers 0.5.0
	pwd
	cd $(TMP_DIR) && cd $(CRATE) && env RUST_LOG=debug $(CARGO) build -j 1 --target $(TARGET) -vv 2>&1 | tee cargo-fail-vv-9.2.log
else

all:

endif
