# Copyright 2014 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

######################################################################
# Distribution
######################################################################

# Primary targets:
#
# * dist - make all distribution artifacts
# * distcheck - sanity check dist artifacts
# * dist-tar-src - source tarballs
# * dist-win - Windows exe installers
# * dist-osx - OS X .pkg installers
# * dist-tar-bins - Ad-hoc Unix binary installers

PKG_NAME = $(CFG_PACKAGE_NAME)

PKG_GITMODULES := $(S)src/libuv $(S)src/llvm $(S)src/gyp $(S)src/compiler-rt

PKG_FILES := \
    $(S)COPYRIGHT                              \
    $(S)LICENSE-APACHE                         \
    $(S)LICENSE-MIT                            \
    $(S)AUTHORS.txt                            \
    $(S)CONTRIBUTING.md                        \
    $(S)README.md                              \
    $(S)RELEASES.txt                           \
    $(S)configure $(S)Makefile.in              \
    $(S)man                                    \
    $(addprefix $(S)src/,                      \
      README.md                                \
      compiletest                              \
      doc                                      \
      driver                                   \
      etc                                      \
      $(foreach crate,$(CRATES),lib$(crate))   \
      rt                                       \
      rustllvm                                 \
      snapshots.txt                            \
      test)                                    \
    $(PKG_GITMODULES)                          \
    $(filter-out Makefile config.stamp config.mk, \
                 $(MKFILE_DEPS))

UNROOTED_PKG_FILES := $(patsubst $(S)%,./%,$(PKG_FILES))

LICENSE.txt: $(S)COPYRIGHT $(S)LICENSE-APACHE $(S)LICENSE-MIT
	cat $^ > $@


######################################################################
# Source tarball
######################################################################

PKG_TAR = dist/$(PKG_NAME).tar.gz

$(PKG_TAR): $(PKG_FILES)
	@$(call E, making dist dir)
	$(Q)rm -Rf tmp/dist/$(PKG_NAME)
	$(Q)mkdir -p tmp/dist/$(PKG_NAME)
	$(Q)tar \
         -C $(S) \
         --exclude-vcs \
         --exclude=*~ \
         --exclude=*/llvm/test/*/*.ll \
         --exclude=*/llvm/test/*/*.td \
         --exclude=*/llvm/test/*/*.s \
         --exclude=*/llvm/test/*/*/*.ll \
         --exclude=*/llvm/test/*/*/*.td \
         --exclude=*/llvm/test/*/*/*.s \
         -c $(UNROOTED_PKG_FILES) | tar -x -C tmp/dist/$(PKG_NAME)
	$(Q)tar -czf $(PKG_TAR) -C tmp/dist $(PKG_NAME)
	$(Q)rm -Rf tmp/dist/$(PKG_NAME)

dist-tar-src: $(PKG_TAR)

######################################################################
# Windows .exe installer
######################################################################

ifdef CFG_ISCC

PKG_EXE = dist/$(PKG_NAME)-install.exe

%.iss: $(S)src/etc/pkg/%.iss
	cp $< $@

%.ico: $(S)src/etc/pkg/%.ico
	cp $< $@

$(PKG_EXE): rust.iss modpath.iss LICENSE.txt rust-logo.ico \
            $(PKG_FILES) $(CSREQ3_T_$(CFG_BUILD)_H_$(CFG_BUILD)) \
            dist-prepare-win
	$(CFG_PYTHON) $(S)src/etc/copy-runtime-deps.py tmp/dist/win/bin
	@$(call E, ISCC: $@)
	$(Q)"$(CFG_ISCC)" $<

dist-prepare-win: PREPARE_HOST=$(CFG_BUILD)
dist-prepare-win: PREPARE_TARGETS=$(CFG_BUILD)
dist-prepare-win: PREPARE_DEST_DIR=tmp/dist/win
dist-prepare-win: PREPARE_DIR_CMD=$(DEFAULT_PREPARE_DIR_CMD)
dist-prepare-win: PREPARE_BIN_CMD=$(DEFAULT_PREPARE_BIN_CMD)
dist-prepare-win: PREPARE_LIB_CMD=$(DEFAULT_PREPARE_LIB_CMD)
dist-prepare-win: PREPARE_MAN_CMD=$(DEFAULT_PREPARE_MAN_CMD)
dist-prepare-win: PREPARE_CLEAN=true
dist-prepare-win: prepare-base

endif

dist-win: $(PKG_EXE)


######################################################################
# OS X .pkg installer
######################################################################

ifeq ($(CFG_OSTYPE), apple-darwin)

define DEF_OSX_PKG
dist-prepare-osx-$(1): PREPARE_HOST=$(1)
dist-prepare-osx-$(1): PREPARE_TARGETS=$(1)
dist-prepare-osx-$(1): PREPARE_DEST_DIR=tmp/dist/pkgroot-$(1)
dist-prepare-osx-$(1): PREPARE_DIR_CMD=$(DEFAULT_PREPARE_DIR_CMD)
dist-prepare-osx-$(1): PREPARE_BIN_CMD=$(DEFAULT_PREPARE_BIN_CMD)
dist-prepare-osx-$(1): PREPARE_LIB_CMD=$(DEFAULT_PREPARE_LIB_CMD)
dist-prepare-osx-$(1): PREPARE_MAN_CMD=$(DEFAULT_PREPARE_MAN_CMD)
dist-prepare-osx-$(1): prepare-base

dist/$(PKG_NAME)-$(1).pkg: $(S)src/etc/pkg/Distribution.xml LICENSE.txt dist-prepare-osx-$(1)
	@$$(call E, making OS X pkg)
	$(Q)pkgbuild --identifier org.rust-lang.rust --root tmp/dist/pkgroot-$(1) rust.pkg
	$(Q)productbuild --distribution $(S)src/etc/pkg/Distribution.xml --resources . dist/$(PKG_NAME)-$(1).pkg
	$(Q)rm -rf tmp rust.pkg

endef

$(foreach host,$(CFG_HOST),$(eval $(call DEF_OSX_PKG,$(host))))

dist-osx: $(foreach host,$(CFG_HOST),dist/$(PKG_NAME)-$(host).pkg)

endif


######################################################################
# Unix binary installer tarballs
######################################################################

dist-install-dirs: $(foreach host,$(CFG_HOST),dist-install-dir-$(host))

dist-tar-bins: $(foreach host,$(CFG_HOST),dist/$(PKG_NAME)-$(host).tar.gz)

define DEF_INSTALLER
dist-install-dir-$(1): PREPARE_HOST=$(1)
dist-install-dir-$(1): PREPARE_TARGETS=$(1)
dist-install-dir-$(1): PREPARE_DEST_DIR=tmp/dist/$$(PKG_NAME)-$(1)
dist-install-dir-$(1): PREPARE_DIR_CMD=$(DEFAULT_PREPARE_DIR_CMD)
dist-install-dir-$(1): PREPARE_BIN_CMD=$(DEFAULT_PREPARE_BIN_CMD)
dist-install-dir-$(1): PREPARE_LIB_CMD=$(DEFAULT_PREPARE_LIB_CMD)
dist-install-dir-$(1): PREPARE_MAN_CMD=$(DEFAULT_PREPARE_MAN_CMD)
dist-install-dir-$(1): PREPARE_CLEAN=true
dist-install-dir-$(1): prepare-base
	$$(Q)(cd $$(PREPARE_DEST_DIR)/ && find -type f) \
      > $$(PREPARE_DEST_DIR)/$$(CFG_LIBDIR_RELATIVE)/$$(CFG_RUSTLIBDIR)/manifest
	$$(Q)$$(PREPARE_MAN_CMD) $$(S)COPYRIGHT $$(PREPARE_DEST_DIR)
	$$(Q)$$(PREPARE_MAN_CMD) $$(S)LICENSE-APACHE $$(PREPARE_DEST_DIR)
	$$(Q)$$(PREPARE_MAN_CMD) $$(S)LICENSE-MIT $$(PREPARE_DEST_DIR)
	$$(Q)$$(PREPARE_MAN_CMD) $$(S)README.md $$(PREPARE_DEST_DIR)
	$$(Q)$$(PREPARE_BIN_CMD) $$(S)src/etc/install.sh $$(PREPARE_DEST_DIR)

dist/$$(PKG_NAME)-$(1).tar.gz: dist-install-dir-$(1)
	@$(call E, build: $$@)
	$$(Q)tar -czf dist/$$(PKG_NAME)-$(1).tar.gz -C tmp/dist $$(PKG_NAME)-$(1)

endef

$(foreach host,$(CFG_HOST),\
  $(eval $(call DEF_INSTALLER,$(host))))


######################################################################
# Primary targets (dist, distcheck)
######################################################################

ifdef CFG_WINDOWSY_$(CFG_BUILD)

dist: dist-win

distcheck: dist
	@echo
	@echo -----------------------------------------------
	@echo Rust ready for distribution (see ./dist)
	@echo -----------------------------------------------

else

dist: dist-tar-src

distcheck: $(PKG_TAR)
	$(Q)rm -Rf dist
	$(Q)mkdir -p dist
	@$(call E, unpacking $(PKG_TAR) in dist/$(PKG_NAME))
	$(Q)cd dist && tar -xzf ../$(PKG_TAR)
	@$(call E, configuring in dist/$(PKG_NAME)-build)
	$(Q)mkdir -p dist/$(PKG_NAME)-build
	$(Q)cd dist/$(PKG_NAME)-build && ../$(PKG_NAME)/configure
	@$(call E, making 'check' in dist/$(PKG_NAME)-build)
	$(Q)+make -C dist/$(PKG_NAME)-build check
	@$(call E, making 'clean' in dist/$(PKG_NAME)-build)
	$(Q)+make -C dist/$(PKG_NAME)-build clean
	$(Q)rm -Rf dist
	@echo
	@echo -----------------------------------------------
	@echo Rust ready for distribution (see ./dist)
	@echo -----------------------------------------------

endif

.PHONY: dist distcheck
