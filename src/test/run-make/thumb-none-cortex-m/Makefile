# Don't use the definitions
#-include ../../run-make-fulldeps/tools.mk

ifeq ($(TARGET),thumbv7m-none-eabi)

#TMP_DIR := $(shell mktemp -d)
TMP_DIR := /tmp/safe_place
CRATE := cortex-m

all:
	env > env.txt
	mkdir -p $(TMP_DIR)
	-cd $(TMP_DIR) && rm -rf $(CRATE)
	cd $(TMP_DIR) && $(CARGO) clone $(CRATE) --vers 0.5.0
	pwd
	cd $(TMP_DIR) && cd $(CRATE) && $(CARGO) build --target $(TARGET) -vv
else

all:

endif
