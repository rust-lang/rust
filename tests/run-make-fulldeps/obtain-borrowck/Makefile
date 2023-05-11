include ../../run-make/tools.mk

# This example shows how to implement a rustc driver that retrieves MIR bodies
# together with the borrow checker information.

# How to run this
# $ ./x.py test tests/run-make-fulldeps/obtain-borrowck

DRIVER_BINARY := "$(TMPDIR)"/driver
SYSROOT := $(shell $(RUSTC) --print sysroot)

ifdef IS_WINDOWS
LIBSTD := -L "$(SYSROOT)\\lib\\rustlib\\$(TARGET)\\lib"
else
LIBSTD :=
endif

all:
	$(RUSTC) driver.rs -o "$(DRIVER_BINARY)"
	$(TARGET_RPATH_ENV) "$(DRIVER_BINARY)" --sysroot $(SYSROOT) $(LIBSTD) test.rs -o "$(TMPDIR)/driver_test" > "$(TMPDIR)"/output.stdout

ifdef RUSTC_BLESS_TEST
	cp "$(TMPDIR)"/output.stdout output.stdout
else
	$(DIFF) output.stdout "$(TMPDIR)"/output.stdout
endif
