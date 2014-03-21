######################################################################
# Distribution
######################################################################

PKG_NAME := rust
PKG_DIR = $(PKG_NAME)-$(CFG_RELEASE)
PKG_TAR = dist/$(PKG_DIR).tar.gz

ifdef CFG_ISCC
PKG_ISS = $(wildcard $(S)src/etc/pkg/*.iss)
PKG_ICO = $(S)src/etc/pkg/rust-logo.ico
PKG_EXE = dist/$(PKG_DIR)-install.exe
endif

ifeq ($(CFG_OSTYPE), apple-darwin)
PKG_OSX = dist/$(PKG_DIR).pkg
endif

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

ifdef CFG_ISCC
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

$(PKG_TAR): $(PKG_FILES)
	@$(call E, making dist dir)
	$(Q)rm -Rf tmp/dist/$(PKG_DIR)
	$(Q)mkdir -p tmp/dist/$(PKG_DIR)
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
         -c $(UNROOTED_PKG_FILES) | tar -x -C tmp/dist/$(PKG_DIR)
	$(Q)tar -czf $(PKG_TAR) -C tmp/dist $(PKG_DIR)
	$(Q)rm -Rf tmp/dist/$(PKG_DIR)

.PHONY: dist distcheck

ifdef CFG_WINDOWSY_$(CFG_BUILD)

dist: $(PKG_EXE)

distcheck: dist
	@echo
	@echo -----------------------------------------------
	@echo $(PKG_EXE) ready for distribution
	@echo -----------------------------------------------

else

dist: $(PKG_TAR) $(PKG_OSX)

distcheck: $(PKG_TAR)
	$(Q)rm -Rf dist
	$(Q)mkdir -p dist
	@$(call E, unpacking $(PKG_TAR) in dist/$(PKG_DIR))
	$(Q)cd dist && tar -xzf ../$(PKG_TAR)
	@$(call E, configuring in dist/$(PKG_DIR)-build)
	$(Q)mkdir -p dist/$(PKG_DIR)-build
	$(Q)cd dist/$(PKG_DIR)-build && ../$(PKG_DIR)/configure
	@$(call E, making 'check' in dist/$(PKG_DIR)-build)
	$(Q)+make -C dist/$(PKG_DIR)-build check
	@$(call E, making 'clean' in dist/$(PKG_DIR)-build)
	$(Q)+make -C dist/$(PKG_DIR)-build clean
	$(Q)rm -Rf dist
	@echo
	@echo -----------------------------------------------
	@echo $(PKG_TAR) ready for distribution
	@echo -----------------------------------------------

endif

ifeq ($(CFG_OSTYPE), apple-darwin)

dist-prepare-osx: PREPARE_HOST=$(CFG_BUILD)
dist-prepare-osx: PREPARE_TARGETS=$(CFG_BUILD)
dist-prepare-osx: PREPARE_DEST_DIR=tmp/dist/pkgroot
dist-prepare-osx: PREPARE_DIR_CMD=$(DEFAULT_PREPARE_DIR_CMD)
dist-prepare-osx: PREPARE_BIN_CMD=$(DEFAULT_PREPARE_BIN_CMD)
dist-prepare-osx: PREPARE_LIB_CMD=$(DEFAULT_PREPARE_LIB_CMD)
dist-prepare-osx: PREPARE_MAN_CMD=$(DEFAULT_PREPARE_MAN_CMD)
dist-prepare-osx: prepare-base

$(PKG_OSX): Distribution.xml LICENSE.txt dist-prepare-osx
	@$(call E, making OS X pkg)
	$(Q)pkgbuild --identifier org.rust-lang.rust --root tmp/dist/pkgroot rust.pkg
	$(Q)productbuild --distribution Distribution.xml --resources . $(PKG_OSX)
	$(Q)rm -rf tmp rust.pkg

dist-osx: $(PKG_OSX)

distcheck-osx: $(PKG_OSX)
	@echo
	@echo -----------------------------------------------
	@echo $(PKG_OSX) ready for distribution
	@echo -----------------------------------------------

endif

dist-install-dir: $(foreach host,$(CFG_HOST),dist-install-dir-$(host))

dist-tar-bins: $(foreach host,$(CFG_HOST),dist/$(PKG_DIR)-$(host).tar.gz)

define DEF_INSTALLER
dist-install-dir-$(1): PREPARE_HOST=$(1)
dist-install-dir-$(1): PREPARE_TARGETS=$(1)
dist-install-dir-$(1): PREPARE_DEST_DIR=tmp/dist/$$(PKG_DIR)-$(1)
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

dist/$$(PKG_DIR)-$(1).tar.gz: dist-install-dir-$(1)
	@$(call E, build: $$@)
	$$(Q)tar -czf dist/$$(PKG_DIR)-$(1).tar.gz -C tmp/dist $$(PKG_DIR)-$(1)

endef

$(foreach host,$(CFG_HOST),\
  $(eval $(call DEF_INSTALLER,$(host))))
