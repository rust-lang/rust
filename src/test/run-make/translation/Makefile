include ../../run-make-fulldeps/tools.mk

# This test uses `ln -s` rather than copying to save testing time, but its
# usage doesn't work on Windows.
# ignore-windows

SYSROOT:=$(shell $(RUSTC) --print sysroot)
FAKEROOT=$(TMPDIR)/fakeroot

all: normal custom sysroot

# Check that the test works normally, using the built-in fallback bundle.
normal: test.rs
	$(RUSTC) $< 2>&1 | grep "struct literal body without path"

# Check that a primary bundle can be loaded and will be preferentially used
# where possible.
custom: test.rs working.ftl
	$(RUSTC) $< -Ztranslate-additional-ftl=$(CURDIR)/working.ftl 2>&1 | grep "this is a test message"

# Check that a primary bundle with a broken message (e.g. a interpolated
# variable is missing) will use the fallback bundle.
missing: test.rs missing.ftl
	$(RUSTC) $< -Ztranslate-additional-ftl=$(CURDIR)/missing.ftl 2>&1 | grep "struct literal body without path"

# Check that a primary bundle without the desired message will use the fallback
# bundle.
broken: test.rs broken.ftl
	$(RUSTC) $< -Ztranslate-additional-ftl=$(CURDIR)/broken.ftl 2>&1 | grep "struct literal body without path"

# Check that a locale can be loaded from the sysroot given a language
# identifier by making a local copy of the sysroot and adding the custom locale
# to it.
sysroot: test.rs working.ftl
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
	ln -s $(CURDIR)/working.ftl $(FAKEROOT)/share/locale/zh-CN/basic-translation.ftl
	$(RUSTC) $< --sysroot $(FAKEROOT) -Ztranslate-lang=zh-CN 2>&1 | grep "this is a test message"

# Check that the compiler errors out when the sysroot requested cannot be
# found. This test might start failing if there actually exists a Klingon
# translation of rustc's error messages.
sysroot-missing: 
	$(RUSTC) $< -Ztranslate-lang=tlh 2>&1 || grep "missing locale directory"

# Check that the compiler errors out when the sysroot requested cannot be
# found. This test might start failing if there actually exists a Klingon
# translation of rustc's error messages.
sysroot-invalid: test.rs working.ftl
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
