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
# * dist-docs - Stage docs for upload

PKG_NAME := $(CFG_PACKAGE_NAME)

# License suitable for displaying in a popup
LICENSE.txt: $(S)COPYRIGHT $(S)LICENSE-APACHE $(S)LICENSE-MIT
	cat $^ > $@


######################################################################
# Source tarball
######################################################################

PKG_TAR = dist/$(PKG_NAME).tar.gz

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
    $(filter-out config.stamp, \
                 $(MKFILES_FOR_TARBALL))

UNROOTED_PKG_FILES := $(patsubst $(S)%,./%,$(PKG_FILES))

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

distcheck-tar-src: dist-tar-src
	$(Q)rm -Rf tmp/distcheck/$(PKG_NAME)
	$(Q)rm -Rf tmp/distcheck/srccheck
	$(Q)mkdir -p tmp/distcheck
	@$(call E, unpacking $(PKG_TAR) in tmp/distcheck/$(PKG_NAME))
	$(Q)cd tmp/distcheck && tar -xzf ../../$(PKG_TAR)
	@$(call E, configuring in tmp/distcheck/srccheck)
	$(Q)mkdir -p tmp/distcheck/srccheck
	$(Q)cd tmp/distcheck/srccheck && ../$(PKG_NAME)/configure
	@$(call E, making 'check' in tmp/distcheck/srccheck)
	$(Q)+make -C tmp/distcheck/srccheck check
	@$(call E, making 'clean' in tmp/distcheck/srccheck)
	$(Q)+make -C tmp/distcheck/srccheck clean
	$(Q)rm -Rf tmp/distcheck/$(PKG_NAME)
	$(Q)rm -Rf tmp/distcheck/srccheck


######################################################################
# Windows .exe installer
######################################################################

# FIXME Needs to support all hosts, but making rust.iss compatible looks like a chore

ifdef CFG_ISCC

PKG_EXE = dist/$(PKG_NAME)-install.exe

%.iss: $(S)src/etc/pkg/%.iss
	cp $< $@

%.ico: $(S)src/etc/pkg/%.ico
	cp $< $@

$(PKG_EXE): rust.iss modpath.iss LICENSE.txt rust-logo.ico \
            $(CSREQ3_T_$(CFG_BUILD)_H_$(CFG_BUILD)) \
            dist-prepare-win
	$(CFG_PYTHON) $(S)src/etc/copy-runtime-deps.py tmp/dist/win/bin
	@$(call E, ISCC: $@)
	$(Q)"$(CFG_ISCC)" $<

$(eval $(call DEF_PREPARE,win))

dist-prepare-win: PREPARE_HOST=$(CFG_BUILD)
dist-prepare-win: PREPARE_TARGETS=$(CFG_BUILD)
dist-prepare-win: PREPARE_DEST_DIR=tmp/dist/win
dist-prepare-win: PREPARE_DIR_CMD=$(DEFAULT_PREPARE_DIR_CMD)
dist-prepare-win: PREPARE_BIN_CMD=$(DEFAULT_PREPARE_BIN_CMD)
dist-prepare-win: PREPARE_LIB_CMD=$(DEFAULT_PREPARE_LIB_CMD)
dist-prepare-win: PREPARE_MAN_CMD=$(DEFAULT_PREPARE_MAN_CMD)
dist-prepare-win: PREPARE_CLEAN=true
dist-prepare-win: prepare-base-win

endif

dist-win: $(PKG_EXE)

distcheck-win: dist-win

######################################################################
# OS X .pkg installer
######################################################################

ifeq ($(CFG_OSTYPE), apple-darwin)

define DEF_OSX_PKG

$$(eval $$(call DEF_PREPARE,osx-$(1)))

dist-prepare-osx-$(1): PREPARE_HOST=$(1)
dist-prepare-osx-$(1): PREPARE_TARGETS=$(1)
dist-prepare-osx-$(1): PREPARE_DEST_DIR=tmp/dist/pkgroot-$(1)
dist-prepare-osx-$(1): PREPARE_DIR_CMD=$(DEFAULT_PREPARE_DIR_CMD)
dist-prepare-osx-$(1): PREPARE_BIN_CMD=$(DEFAULT_PREPARE_BIN_CMD)
dist-prepare-osx-$(1): PREPARE_LIB_CMD=$(DEFAULT_PREPARE_LIB_CMD)
dist-prepare-osx-$(1): PREPARE_MAN_CMD=$(DEFAULT_PREPARE_MAN_CMD)
dist-prepare-osx-$(1): prepare-base-osx-$(1)

dist/$(PKG_NAME)-$(1).pkg: $(S)src/etc/pkg/Distribution.xml LICENSE.txt \
		dist-prepare-osx-$(1) \
		tmp/dist/pkgres-$(1)/LICENSE.txt \
		tmp/dist/pkgres-$(1)/welcome.rtf \
		tmp/dist/pkgres-$(1)/rust-logo.png
	@$$(call E, making OS X pkg)
	$(Q)pkgbuild --identifier org.rust-lang.rust --root tmp/dist/pkgroot-$(1) rust.pkg
	$(Q)productbuild --distribution $(S)src/etc/pkg/Distribution.xml \
	      --resources tmp/dist/pkgres-$(1) dist/$(PKG_NAME)-$(1).pkg
	$(Q)rm -rf tmp rust.pkg

tmp/dist/pkgres-$(1)/LICENSE.txt: LICENSE.txt
	@$$(call E,pkg resource LICENSE.txt)
	$(Q)mkdir -p $$(@D)
	$(Q)cp $$< $$@

tmp/dist/pkgres-$(1)/%: $(S)src/etc/pkg/%
	@$$(call E,pkg resource $$*)
	$(Q)mkdir -p $$(@D)
	$(Q)cp -r $$< $$@

endef

$(foreach host,$(CFG_HOST),$(eval $(call DEF_OSX_PKG,$(host))))

dist-osx: $(foreach host,$(CFG_HOST),dist/$(PKG_NAME)-$(host).pkg)

else

dist-osx:

endif

# FIXME should do something
distcheck-osx: dist-osx


######################################################################
# Unix binary installer tarballs
######################################################################

define DEF_INSTALLER

$$(eval $$(call DEF_PREPARE,dir-$(1)))

dist-install-dir-$(1): PREPARE_HOST=$(1)
dist-install-dir-$(1): PREPARE_TARGETS=$(1)
dist-install-dir-$(1): PREPARE_DEST_DIR=tmp/dist/$$(PKG_NAME)-$(1)
dist-install-dir-$(1): PREPARE_DIR_CMD=$(DEFAULT_PREPARE_DIR_CMD)
dist-install-dir-$(1): PREPARE_BIN_CMD=$(DEFAULT_PREPARE_BIN_CMD)
dist-install-dir-$(1): PREPARE_LIB_CMD=$(DEFAULT_PREPARE_LIB_CMD)
dist-install-dir-$(1): PREPARE_MAN_CMD=$(DEFAULT_PREPARE_MAN_CMD)
dist-install-dir-$(1): PREPARE_CLEAN=true
dist-install-dir-$(1): prepare-base-dir-$(1)
	$$(Q)(cd $$(PREPARE_DEST_DIR)/ && find . -type f | sed 's/^\.\///') \
      > tmp/dist/manifest-$(1).in
	$$(Q)mv tmp/dist/manifest-$(1).in $$(PREPARE_DEST_DIR)/$$(CFG_LIBDIR_RELATIVE)/rustlib/manifest.in
# Add remaining non-installed files
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

dist-install-dirs: $(foreach host,$(CFG_HOST),dist-install-dir-$(host))

dist-tar-bins: $(foreach host,$(CFG_HOST),dist/$(PKG_NAME)-$(host).tar.gz)

# Just try to run the compiler for the build host
distcheck-tar-bins: dist-tar-bins
	@$(call E, checking binary tarball)
	$(Q)rm -Rf tmp/distcheck/$(PKG_NAME)-$(CFG_BUILD)
	$(Q)rm -Rf tmp/distcheck/tarbininstall
	$(Q)mkdir -p tmp/distcheck
	$(Q)cd tmp/distcheck && tar -xzf ../../dist/$(PKG_NAME)-$(CFG_BUILD).tar.gz
	$(Q)mkdir -p tmp/distcheck/tarbininstall
	$(Q)sh tmp/distcheck/$(PKG_NAME)-$(CFG_BUILD)/install.sh --prefix=tmp/distcheck/tarbininstall
	$(Q)tmp/distcheck/tarbininstall/bin/rustc --version
	$(Q)rm -Rf tmp/distcheck/$(PKG_NAME)-$(CFG_BUILD)
	$(Q)rm -Rf tmp/distcheck/tarbininstall

######################################################################
# Docs
######################################################################

# Just copy the docs to a folder under dist with the appropriate name
# for uploading to S3
dist-docs: docs compiler-docs
	$(Q) rm -Rf dist/doc
	$(Q) mkdir -p dist/doc/
	$(Q) cp -r doc dist/doc/$(CFG_PACKAGE_VERS)

distcheck-docs: dist-docs

######################################################################
# Primary targets (dist, distcheck)
######################################################################

ifdef CFG_WINDOWSY_$(CFG_BUILD)

dist: dist-win

distcheck: distcheck-win
	$(Q)rm -Rf tmp/distcheck
	@echo
	@echo -----------------------------------------------
	@echo "Rust ready for distribution (see ./dist)"
	@echo -----------------------------------------------

else

dist: dist-tar-src dist-osx dist-tar-bins dist-docs

distcheck: distcheck-tar-src distcheck-osx distcheck-tar-bins distcheck-docs
	$(Q)rm -Rf tmp/distcheck
	@echo
	@echo -----------------------------------------------
	@echo "Rust ready for distribution (see ./dist)"
	@echo -----------------------------------------------

endif

.PHONY: dist distcheck
