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
# * dist-tar-bins - Ad-hoc Unix binary installers
# * dist-docs - Stage docs for upload

PKG_NAME := $(CFG_PACKAGE_NAME)
DOC_PKG_NAME := rust-docs-$(CFG_PACKAGE_VERS)
MINGW_PKG_NAME := rust-mingw-$(CFG_PACKAGE_VERS)

# License suitable for displaying in a popup
LICENSE.txt: $(S)COPYRIGHT $(S)LICENSE-APACHE $(S)LICENSE-MIT
	cat $^ > $@


######################################################################
# Source tarball
######################################################################

PKG_TAR = dist/$(PKG_NAME)-src.tar.gz

PKG_GITMODULES := $(S)src/llvm $(S)src/compiler-rt \
		  $(S)src/rt/hoedown $(S)src/jemalloc
PKG_FILES := \
    $(S)COPYRIGHT                              \
    $(S)LICENSE-APACHE                         \
    $(S)LICENSE-MIT                            \
    $(S)AUTHORS.txt                            \
    $(S)CONTRIBUTING.md                        \
    $(S)README.md                              \
    $(S)RELEASES.md                            \
    $(S)configure $(S)Makefile.in              \
    $(S)man                                    \
    $(addprefix $(S)src/,                      \
      compiletest                              \
      doc                                      \
      driver                                   \
      etc                                      \
      $(foreach crate,$(CRATES),lib$(crate))   \
      libcoretest                              \
      libbacktrace                             \
      rt                                       \
      rustllvm                                 \
      snapshots.txt                            \
      rust-installer                           \
      rustbook                                 \
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
	@$(call E, making $@)
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
# Unix binary installer tarballs
######################################################################

NON_INSTALLED_PREFIXES=COPYRIGHT,LICENSE-APACHE,LICENSE-MIT,README.md,version

define DEF_INSTALLER

$$(eval $$(call DEF_PREPARE,dir-$(1)))

dist-install-dir-$(1): PREPARE_HOST=$(1)
dist-install-dir-$(1): PREPARE_TARGETS=$(2)
dist-install-dir-$(1): PREPARE_DEST_DIR=tmp/dist/$$(PKG_NAME)-$(1)-image
dist-install-dir-$(1): PREPARE_DIR_CMD=$(DEFAULT_PREPARE_DIR_CMD)
dist-install-dir-$(1): PREPARE_BIN_CMD=$(DEFAULT_PREPARE_BIN_CMD)
dist-install-dir-$(1): PREPARE_LIB_CMD=$(DEFAULT_PREPARE_LIB_CMD)
dist-install-dir-$(1): PREPARE_MAN_CMD=$(DEFAULT_PREPARE_MAN_CMD)
dist-install-dir-$(1): PREPARE_CLEAN=true
dist-install-dir-$(1): prepare-base-dir-$(1) docs compiler-docs
	$$(Q)$$(PREPARE_MAN_CMD) $$(S)COPYRIGHT $$(PREPARE_DEST_DIR)
	$$(Q)$$(PREPARE_MAN_CMD) $$(S)LICENSE-APACHE $$(PREPARE_DEST_DIR)
	$$(Q)$$(PREPARE_MAN_CMD) $$(S)LICENSE-MIT $$(PREPARE_DEST_DIR)
	$$(Q)$$(PREPARE_MAN_CMD) $$(S)README.md $$(PREPARE_DEST_DIR)
	$$(Q)mkdir -p $$(PREPARE_DEST_DIR)/share/doc/rust
	$$(Q)$$(PREPARE_MAN_CMD) $$(S)COPYRIGHT $$(PREPARE_DEST_DIR)/share/doc/rust
	$$(Q)$$(PREPARE_MAN_CMD) $$(S)LICENSE-APACHE $$(PREPARE_DEST_DIR)/share/doc/rust
	$$(Q)$$(PREPARE_MAN_CMD) $$(S)LICENSE-MIT $$(PREPARE_DEST_DIR)/share/doc/rust
	$$(Q)$$(PREPARE_MAN_CMD) $$(S)README.md $$(PREPARE_DEST_DIR)/share/doc/rust
# This tiny morsel of metadata is used by rust-packaging
	$$(Q)echo "$(CFG_VERSION)" > $$(PREPARE_DEST_DIR)/version

dist/$$(PKG_NAME)-$(1).tar.gz: dist-install-dir-$(1)
	@$(call E, build: $$@)
# Copy essential gcc components into installer
ifdef CFG_WINDOWSY_$(1)
	$$(Q)rm -Rf tmp/dist/win-rust-gcc-$(1)
	$$(Q)$$(CFG_PYTHON) $$(S)src/etc/make-win-dist.py tmp/dist/$$(PKG_NAME)-$(1)-image tmp/dist/win-rust-gcc-$(1) $(1)
	$$(Q)cp -r $$(S)src/etc/third-party tmp/dist/$$(PKG_NAME)-$(1)-image/share/doc/
endif
	$$(Q)$$(S)src/rust-installer/gen-installer.sh \
		--product-name=Rust \
		--verify-bin=rustc \
		--rel-manifest-dir=rustlib \
		--success-message=Rust-is-ready-to-roll. \
		--image-dir=tmp/dist/$$(PKG_NAME)-$(1)-image \
		--work-dir=tmp/dist \
		--output-dir=dist \
		--non-installed-prefixes=$$(NON_INSTALLED_PREFIXES) \
		--package-name=$$(PKG_NAME)-$(1) \
		--component-name=rustc \
		--legacy-manifest-dirs=rustlib,cargo
	$$(Q)rm -R tmp/dist/$$(PKG_NAME)-$(1)-image

dist-doc-install-dir-$(1): docs compiler-docs
	$$(Q)mkdir -p tmp/dist/$$(DOC_PKG_NAME)-$(1)-image/share/doc/rust
	$$(Q)cp -r doc tmp/dist/$$(DOC_PKG_NAME)-$(1)-image/share/doc/rust/html

dist/$$(DOC_PKG_NAME)-$(1).tar.gz: dist-doc-install-dir-$(1)
	@$(call E, build: $$@)
	$$(Q)$$(S)src/rust-installer/gen-installer.sh \
		--product-name=Rust-Documentation \
		--rel-manifest-dir=rustlib \
		--success-message=Rust-documentation-is-installed. \
		--image-dir=tmp/dist/$$(DOC_PKG_NAME)-$(1)-image \
		--work-dir=tmp/dist \
		--output-dir=dist \
		--package-name=$$(DOC_PKG_NAME)-$(1) \
		--component-name=rust-docs \
		--legacy-manifest-dirs=rustlib,cargo \
		--bulk-dirs=share/doc/rust/html
	$$(Q)rm -R tmp/dist/$$(DOC_PKG_NAME)-$(1)-image

dist-mingw-install-dir-$(1):
	$$(Q)mkdir -p tmp/dist/rust-mingw-tmp-$(1)-image
	$$(Q)rm -Rf tmp/dist/$$(MINGW_PKG_NAME)-$(1)-image
	$$(Q)$$(CFG_PYTHON) $$(S)src/etc/make-win-dist.py \
		tmp/dist/rust-mingw-tmp-$(1)-image tmp/dist/$$(MINGW_PKG_NAME)-$(1)-image $(1)

dist/$$(MINGW_PKG_NAME)-$(1).tar.gz: dist-mingw-install-dir-$(1)
	@$(call E, build: $$@)
	$$(Q)$$(S)src/rust-installer/gen-installer.sh \
		--product-name=Rust-MinGW \
		--rel-manifest-dir=rustlib \
		--success-message=Rust-MinGW-is-installed. \
		--image-dir=tmp/dist/$$(MINGW_PKG_NAME)-$(1)-image \
		--work-dir=tmp/dist \
		--output-dir=dist \
		--package-name=$$(MINGW_PKG_NAME)-$(1) \
		--component-name=rust-mingw \
		--legacy-manifest-dirs=rustlib,cargo
	$$(Q)rm -R tmp/dist/$$(MINGW_PKG_NAME)-$(1)-image

endef

ifneq ($(CFG_ENABLE_DIST_HOST_ONLY),)
$(foreach host,$(CFG_HOST),\
  $(eval $(call DEF_INSTALLER,$(host),$(host))))
else
$(foreach host,$(CFG_HOST),\
  $(eval $(call DEF_INSTALLER,$(host),$(CFG_TARGET))))
endif

dist-install-dirs: $(foreach host,$(CFG_HOST),dist-install-dir-$(host))

ifdef CFG_WINDOWSY_$(CFG_BUILD)
MAYBE_MINGW_TARBALLS=$(foreach host,$(CFG_HOST),dist/$(MINGW_PKG_NAME)-$(host).tar.gz)
endif

ifeq ($(CFG_DISABLE_DOCS),)
MAYBE_DOC_TARBALLS=$(foreach host,$(CFG_HOST),dist/$(DOC_PKG_NAME)-$(host).tar.gz)
endif

dist-tar-bins: $(foreach host,$(CFG_HOST),dist/$(PKG_NAME)-$(host).tar.gz) \
	$(MAYBE_DOC_TARBALLS) $(MAYBE_MINGW_TARBALLS)

# Just try to run the compiler for the build host
distcheck-tar-bins: dist-tar-bins
	@$(call E, checking binary tarball)
	$(Q)rm -Rf tmp/distcheck/$(PKG_NAME)-$(CFG_BUILD)
	$(Q)rm -Rf tmp/distcheck/tarbininstall
	$(Q)mkdir -p tmp/distcheck
	$(Q)cd tmp/distcheck && tar -xzf ../../dist/$(PKG_NAME)-$(CFG_BUILD).tar.gz
	$(Q)mkdir -p tmp/distcheck/tarbininstall
	$(Q)sh tmp/distcheck/$(PKG_NAME)-$(CFG_BUILD)/install.sh --prefix=tmp/distcheck/tarbininstall
	$(Q)sh tmp/distcheck/$(PKG_NAME)-$(CFG_BUILD)/install.sh --prefix=tmp/distcheck/tarbininstall --uninstall
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

MAYBE_DIST_TAR_SRC=dist-tar-src
MAYBE_DISTCHECK_TAR_SRC=distcheck-tar-src

# FIXME #13224: On OS X don't produce tarballs simply because --exclude-vcs don't work.
# This is a huge hack because I just don't have time to figure out another solution.
ifeq ($(CFG_OSTYPE), apple-darwin)
MAYBE_DIST_TAR_SRC=
MAYBE_DISTCHECK_TAR_SRC=
endif

# Don't bother with source tarballs on windows just because we historically haven't.
ifeq ($(CFG_OSTYPE), pc-windows-gnu)
MAYBE_DIST_TAR_SRC=
MAYBE_DISTCHECK_TAR_SRC=
endif

ifneq ($(CFG_DISABLE_DOCS),)
MAYBE_DIST_DOCS=
MAYBE_DISTCHECK_DOCS=
else
MAYBE_DIST_DOCS=dist-docs
MAYBE_DISTCHECK_DOCS=distcheck-docs
endif

dist: $(MAYBE_DIST_TAR_SRC) dist-tar-bins $(MAYBE_DIST_DOCS)

distcheck: $(MAYBE_DISTCHECK_TAR_SRC) distcheck-tar-bins $(MAYBE_DISTCHECK_DOCS)
	$(Q)rm -Rf tmp/distcheck
	@echo
	@echo -----------------------------------------------
	@echo "Rust ready for distribution (see ./dist)"
	@echo -----------------------------------------------

.PHONY: dist distcheck
