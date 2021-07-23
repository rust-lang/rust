include ../tools.mk

# This example shows how to implement a rustc driver that retrieves MIR bodies
# together with the borrow checker information.

# How to run this
# $ ./x.py test src/test/run-make/obtain-borrowck

DRIVER_BINARY := "$(TMPDIR)"/driver
SYSROOT := $(shell $(RUSTC) --print sysroot)

all:
	$(RUSTC) driver.rs -o "$(DRIVER_BINARY)"
	$(TARGET_RPATH_ENV) "$(DRIVER_BINARY)" --sysroot $(SYSROOT) test.rs -o "$(TMPDIR)/driver_test" > "$(TMPDIR)"/output.stdout

ifdef RUSTC_BLESS_TEST
	cp "$(TMPDIR)"/output.stdout output.stdout
else
	$(DIFF) output.stdout "$(TMPDIR)"/output.stdout
endif
