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
STD_PKG_NAME := rust-std-$(CFG_PACKAGE_VERS)
DOC_PKG_NAME := rust-docs-$(CFG_PACKAGE_VERS)
MINGW_PKG_NAME := rust-mingw-$(CFG_PACKAGE_VERS)
SRC_PKG_NAME := rust-src-$(CFG_PACKAGE_VERS)

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
    $(S)CONTRIBUTING.md                        \
    $(S)README.md                              \
    $(S)RELEASES.md                            \
    $(S)configure $(S)Makefile.in              \
    $(S)man                                    \
    $(addprefix $(S)src/,                      \
      bootstrap                                \
      build_helper                             \
      doc                                      \
      driver                                   \
      etc                                      \
      $(foreach crate,$(CRATES),lib$(crate))   \
      libcollectionstest                       \
      libcoretest                              \
      libbacktrace                             \
      rt                                       \
      rtstartup                                \
      rustllvm                                 \
      rustc                                    \
      stage0.txt                               \
      rust-installer                           \
      tools                                    \
      test                                     \
      vendor)                                  \
    $(PKG_GITMODULES)                          \
    $(filter-out config.stamp, \
                 $(MKFILES_FOR_TARBALL))

UNROOTED_PKG_FILES := $(patsubst $(S)%,./%,$(PKG_FILES))

tmp/dist/$$(SRC_PKG_NAME)-image: $(PKG_FILES)
	@$(call E, making src image)
	$(Q)rm -Rf tmp/dist/$(SRC_PKG_NAME)-image
	$(Q)mkdir -p tmp/dist/$(SRC_PKG_NAME)-image/lib/rustlib/src/rust
	$(Q)echo "$(CFG_VERSION)" > tmp/dist/$(SRC_PKG_NAME)-image/lib/rustlib/src/rust/version
	$(Q)tar \
         -C $(S) \
         -f - \
         --exclude-vcs \
         --exclude=*~ \
         --exclude=*.pyc \
         --exclude=*/llvm/test/*/*.ll \
         --exclude=*/llvm/test/*/*.td \
         --exclude=*/llvm/test/*/*.s \
         --exclude=*/llvm/test/*/*/*.ll \
         --exclude=*/llvm/test/*/*/*.td \
         --exclude=*/llvm/test/*/*/*.s \
         -c $(UNROOTED_PKG_FILES) | tar -x -f - -C tmp/dist/$(SRC_PKG_NAME)-image/lib/rustlib/src/rust

$(PKG_TAR): tmp/dist/$$(SRC_PKG_NAME)-image
	@$(call E, making $@)
	$(Q)tar -czf $(PKG_TAR) -C tmp/dist/$(SRC_PKG_NAME)-image/lib/rustlib/src rust --transform 's,^rust,$(PKG_NAME),S'

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

define DEF_START_INSTALLER
dist-install-dir-$(1)-%: PREPARE_DIR_CMD=$(DEFAULT_PREPARE_DIR_CMD)
dist-install-dir-$(1)-%: PREPARE_BIN_CMD=$(DEFAULT_PREPARE_BIN_CMD)
dist-install-dir-$(1)-%: PREPARE_LIB_CMD=$(DEFAULT_PREPARE_LIB_CMD)
dist-install-dir-$(1)-%: PREPARE_MAN_CMD=$(DEFAULT_PREPARE_MAN_CMD)
dist-install-dir-$(1)-%: PREPARE_CLEAN=true

$$(eval $$(call DEF_PREPARE,dir-$(1)))
endef

$(foreach target,$(CFG_TARGET),\
  $(eval $(call DEF_START_INSTALLER,$(target))))

define DEF_INSTALLER

dist-install-dir-$(1)-host: PREPARE_HOST=$(1)
dist-install-dir-$(1)-host: PREPARE_TARGETS=$(2)
dist-install-dir-$(1)-host: PREPARE_DEST_DIR=tmp/dist/$$(PKG_NAME)-$(1)-image
dist-install-dir-$(1)-host: prepare-base-dir-$(1)-host docs
	$$(Q)mkdir -p $$(PREPARE_DEST_DIR)/share/doc/rust
	$$(Q)$$(PREPARE_MAN_CMD) $$(S)COPYRIGHT $$(PREPARE_DEST_DIR)/share/doc/rust
	$$(Q)$$(PREPARE_MAN_CMD) $$(S)LICENSE-APACHE $$(PREPARE_DEST_DIR)/share/doc/rust
	$$(Q)$$(PREPARE_MAN_CMD) $$(S)LICENSE-MIT $$(PREPARE_DEST_DIR)/share/doc/rust
	$$(Q)$$(PREPARE_MAN_CMD) $$(S)README.md $$(PREPARE_DEST_DIR)/share/doc/rust

prepare-overlay-$(1):
	$$(Q)rm -Rf tmp/dist/$$(PKG_NAME)-$(1)-overlay
	$$(Q)mkdir -p tmp/dist/$$(PKG_NAME)-$(1)-overlay
	$$(Q)cp $$(S)COPYRIGHT tmp/dist/$$(PKG_NAME)-$(1)-overlay/
	$$(Q)cp $$(S)LICENSE-APACHE tmp/dist/$$(PKG_NAME)-$(1)-overlay/
	$$(Q)cp $$(S)LICENSE-MIT tmp/dist/$$(PKG_NAME)-$(1)-overlay/
	$$(Q)cp $$(S)README.md tmp/dist/$$(PKG_NAME)-$(1)-overlay/
# This tiny morsel of metadata is used by rust-packaging
	$$(Q)echo "$(CFG_VERSION)" > tmp/dist/$$(PKG_NAME)-$(1)-overlay/version

dist/$$(PKG_NAME)-$(1).tar.gz: dist-install-dir-$(1)-host prepare-overlay-$(1)
	@$(call E, build: $$@)
# On a MinGW target we've got a few runtime DLL dependencies that we need
# to include. THe first argument to `make-win-dist` is where to put these DLLs
# (the image we're creating) and the second argument is a junk directory to
# ignore all the other MinGW stuff the script creates.
ifeq ($$(findstring pc-windows-gnu,$(1)),pc-windows-gnu)
	$$(Q)rm -Rf tmp/dist/win-rust-gcc-$(1)
	$$(Q)$$(CFG_PYTHON) $$(S)src/etc/make-win-dist.py \
		tmp/dist/$$(PKG_NAME)-$(1)-image \
		tmp/dist/win-rust-gcc-$(1) $(1)
endif
# On 32-bit MinGW we're always including a DLL which needs some extra licenses
# to distribute. On 64-bit MinGW we don't actually distribute anything requiring
# us to distribute a license but it's likely that the install will *also*
# include the rust-mingw package down below, which also need licenses, so to be
# safe we just inlude it here in all MinGW packages.
ifdef CFG_WINDOWSY_$(1)
ifeq ($$(findstring $(1),gnu),gnu)
	$$(Q)cp -r $$(S)src/etc/third-party \
		tmp/dist/$$(PKG_NAME)-$(1)-image/share/doc/
endif
endif
	$$(Q)$$(S)src/rust-installer/gen-installer.sh \
		--product-name=Rust \
		--rel-manifest-dir=rustlib \
		--success-message=Rust-is-ready-to-roll. \
		--image-dir=tmp/dist/$$(PKG_NAME)-$(1)-image \
		--work-dir=tmp/dist \
		--output-dir=dist \
		--non-installed-overlay=tmp/dist/$$(PKG_NAME)-$(1)-overlay \
		--package-name=$$(PKG_NAME)-$(1) \
		--component-name=rustc \
		--legacy-manifest-dirs=rustlib,cargo
	$$(Q)rm -R tmp/dist/$$(PKG_NAME)-$(1)-image

dist-doc-install-dir-$(1): docs
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

# Creates the rust-mingw package, and the first argument to make-win-dist is a
# "temporary directory" which is just thrown away (this contains the runtime
# DLLs included in the rustc package above) and the second argument is where to
# place all the MinGW components (which is what we want).
dist-mingw-install-dir-$(1):
	$$(Q)mkdir -p tmp/dist/rust-mingw-tmp-$(1)-image
	$$(Q)rm -Rf tmp/dist/$$(MINGW_PKG_NAME)-$(1)-image
	$$(Q)$$(CFG_PYTHON) $$(S)src/etc/make-win-dist.py \
		tmp/dist/rust-mingw-tmp-$(1)-image \
		tmp/dist/$$(MINGW_PKG_NAME)-$(1)-image $(1)

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

# $(1) - host
# $(2) - target
define DEF_INSTALLER_TARGETS

dist-install-dir-$(2)-target: PREPARE_HOST=$(1)
dist-install-dir-$(2)-target: PREPARE_TARGETS=$(2)
dist-install-dir-$(2)-target: PREPARE_DEST_DIR=tmp/dist/$$(STD_PKG_NAME)-$(2)-image
dist-install-dir-$(2)-target: prepare-base-dir-$(2)-target

dist/$$(STD_PKG_NAME)-$(2).tar.gz: dist-install-dir-$(2)-target
	@$$(call E, build: $$@)
	$$(Q)$$(S)src/rust-installer/gen-installer.sh \
		--product-name=Rust \
		--rel-manifest-dir=rustlib \
		--success-message=std-is-standing-at-the-ready. \
		--image-dir=tmp/dist/$$(STD_PKG_NAME)-$(2)-image \
		--work-dir=tmp/dist \
		--output-dir=dist \
		--package-name=$$(STD_PKG_NAME)-$(2) \
		--component-name=rust-std-$(2) \
		--legacy-manifest-dirs=rustlib,cargo
	$$(Q)rm -R tmp/dist/$$(STD_PKG_NAME)-$(2)-image
endef

$(foreach host,$(CFG_HOST),\
  $(eval $(call DEF_INSTALLER,$(host))))

dist/$(SRC_PKG_NAME).tar.gz: tmp/dist/$(SRC_PKG_NAME)-image
	@$(call E, build: $@)
	$(Q)$(S)src/rust-installer/gen-installer.sh \
		--product-name=Rust \
		--rel-manifest-dir=rustlib \
		--success-message=Awesome-Source. \
		--image-dir=tmp/dist/$(SRC_PKG_NAME)-image \
		--work-dir=tmp/dist \
		--output-dir=dist \
		--package-name=$(SRC_PKG_NAME) \
		--component-name=rust-src \
		--legacy-manifest-dirs=rustlib,cargo

# When generating packages for the standard library, we've actually got a lot of
# artifacts to choose from. Each of the CFG_HOST compilers will have a copy of
# the standard library for each CFG_TARGET, but we only want to generate one
# standard library package. As a result, for each entry in CFG_TARGET we need to
# pick a CFG_HOST to get the standard library from.
#
# In theory it doesn't actually matter what host we choose as it should be the
# case that all hosts produce the same set of libraries for a target (regardless
# of the host itself). Currently there is a bug in the compiler, however, which
# means this is not the case (see #29228 and #29235). To solve the first of
# those bugs, we prefer to select a standard library from the host it was
# generated from, allowing plugins to work in more situations.
#
# For all CFG_TARGET entries in CFG_HOST, however, we just pick CFG_BUILD as the
# host we slurp up a standard library from.
$(foreach host,$(CFG_HOST),\
  $(eval $(call DEF_INSTALLER_TARGETS,$(host),$(host))))
$(foreach target,$(filter-out $(CFG_HOST),$(CFG_TARGET)),\
  $(eval $(call DEF_INSTALLER_TARGETS,$(CFG_BUILD),$(target))))

ifdef CFG_WINDOWSY_$(CFG_BUILD)
define BUILD_MINGW_TARBALL
ifeq ($$(findstring gnu,$(1)),gnu)
MAYBE_MINGW_TARBALLS += dist/$(MINGW_PKG_NAME)-$(1).tar.gz
endif
endef

$(foreach host,$(CFG_HOST),\
  $(eval $(call BUILD_MINGW_TARBALL,$(host))))
endif

ifeq ($(CFG_DISABLE_DOCS),)
MAYBE_DOC_TARBALLS=$(foreach host,$(CFG_HOST),dist/$(DOC_PKG_NAME)-$(host).tar.gz)
endif

dist-tar-bins: \
	$(foreach host,$(CFG_HOST),dist/$(PKG_NAME)-$(host).tar.gz) \
	$(foreach target,$(CFG_TARGET),dist/$(STD_PKG_NAME)-$(target).tar.gz) \
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
dist-docs: docs
	$(Q) rm -Rf dist/doc
	$(Q) mkdir -p dist/doc/
	$(Q) cp -r doc dist/doc/$(CFG_PACKAGE_VERS)

distcheck-docs: dist-docs

######################################################################
# Primary targets (dist, distcheck)
######################################################################

MAYBE_DIST_TAR_SRC=dist-tar-src dist/$(SRC_PKG_NAME).tar.gz
MAYBE_DISTCHECK_TAR_SRC=distcheck-tar-src dist/$(SRC_PKG_NAME).tar.gz

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
