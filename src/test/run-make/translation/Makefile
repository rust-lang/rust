include ../../run-make-fulldeps/tools.mk

# This test uses `ln -s` rather than copying to save testing time, but its
# usage doesn't work on Windows.
# ignore-windows

SYSROOT:=$(shell $(RUSTC) --print sysroot)
FAKEROOT=$(TMPDIR)/fakeroot

all: normal custom sysroot

normal: basic-translation.rs
	$(RUSTC) $< 2>&1 | grep "struct literal body without path"

custom: basic-translation.rs basic-translation.ftl
	$(RUSTC) $< -Ztranslate-additional-ftl=$(CURDIR)/basic-translation.ftl 2>&1 | grep "this is a test message"

# Check that a locale can be loaded from the sysroot given a language
# identifier by making a local copy of the sysroot and adding the custom locale
# to it.
sysroot: basic-translation.rs basic-translation.ftl
	mkdir $(FAKEROOT)
	ln -s $(SYSROOT)/* $(FAKEROOT)
	rm -f $(FAKEROOT)/lib
	mkdir $(FAKEROOT)/lib
	ln -s $(SYSROOT)/lib/* $(FAKEROOT)/lib
	rm -f $(FAKEROOT)/lib/rustlib
	mkdir $(FAKEROOT)/lib/rustlib
	ln -s $(SYSROOT)/lib/rustlib/* $(FAKEROOT)/lib/rustlib
	rm -f $(FAKEROOT)/lib/rustlib/src
	mkdir $(FAKEROOT)/lib/rustlib/src
	ln -s $(SYSROOT)/lib/rustlib/src/* $(FAKEROOT)/lib/rustlib/src
	mkdir -p $(FAKEROOT)/share/locale/zh-CN/
	ln -s $(CURDIR)/basic-translation.ftl $(FAKEROOT)/share/locale/zh-CN/basic-translation.ftl
	$(RUSTC) $< --sysroot $(FAKEROOT) -Ztranslate-lang=zh-CN 2>&1 | grep "this is a test message"

# Check that the compiler errors out when the sysroot requested cannot be
# found. This test might start failing if there actually exists a Klingon
# translation of rustc's error messages.
sysroot-missing: 
	$(RUSTC) $< -Ztranslate-lang=tlh 2>&1 || grep "missing locale directory"

# Check that the compiler errors out when the sysroot requested cannot be
# found. This test might start failing if there actually exists a Klingon
# translation of rustc's error messages.
sysroot-invalid: basic-translation.rs basic-translation.ftl
	mkdir $(FAKEROOT)
	ln -s $(SYSROOT)/* $(FAKEROOT)
	rm -f $(FAKEROOT)/lib
	mkdir $(FAKEROOT)/lib
	ln -s $(SYSROOT)/lib/* $(FAKEROOT)/lib
	rm -f $(FAKEROOT)/lib/rustlib
	mkdir $(FAKEROOT)/lib/rustlib
	ln -s $(SYSROOT)/lib/rustlib/* $(FAKEROOT)/lib/rustlib
	rm -f $(FAKEROOT)/lib/rustlib/src
	mkdir $(FAKEROOT)/lib/rustlib/src
	ln -s $(SYSROOT)/lib/rustlib/src/* $(FAKEROOT)/lib/rustlib/src
	touch $(FAKEROOT)/share/locale/zh-CN/
	$(RUSTC) $< --sysroot $(FAKEROOT) -Ztranslate-lang=zh-CN 2>&1 || grep "`\$sysroot/share/locales/\$locale` is not a directory"
