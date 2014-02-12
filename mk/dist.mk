######################################################################
# Distribution
######################################################################

PKG_NAME := rust
PKG_DIR = $(PKG_NAME)-$(CFG_RELEASE)
PKG_TAR = $(PKG_DIR).tar.gz

ifdef CFG_ISCC
PKG_ISS = $(wildcard $(S)src/etc/pkg/*.iss)
PKG_ICO = $(S)src/etc/pkg/rust-logo.ico
PKG_EXE = $(PKG_DIR)-install.exe
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
    $(S)doc                                    \
    $(addprefix $(S)src/,                      \
      README.md                                \
      compiletest                              \
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

ifdef CFG_ISCC
LICENSE.txt: $(S)COPYRIGHT $(S)LICENSE-APACHE $(S)LICENSE-MIT
	cat $^ > $@

%.iss: $(S)src/etc/pkg/%.iss
	cp $< $@

%.ico: $(S)src/etc/pkg/%.ico
	cp $< $@

$(PKG_EXE): rust.iss modpath.iss LICENSE.txt rust-logo.ico \
            $(PKG_FILES) $(CSREQ3_T_$(CFG_BUILD)_H_$(CFG_BUILD))
	$(CFG_PYTHON) $(S)src/etc/copy-runtime-deps.py i686-pc-mingw32/stage3/bin
	@$(call E, ISCC: $@)
	$(Q)"$(CFG_ISCC)" $<
endif


$(PKG_TAR): $(PKG_FILES)
	@$(call E, making dist dir)
	$(Q)rm -Rf dist
	$(Q)mkdir -p dist/$(PKG_DIR)
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
         -c $(UNROOTED_PKG_FILES) | tar -x -C dist/$(PKG_DIR)
	$(Q)tar -czf $(PKG_TAR) -C dist $(PKG_DIR)
	$(Q)rm -Rf dist

.PHONY: dist distcheck

ifdef CFG_WINDOWSY_$(CFG_BUILD)

dist: $(PKG_EXE)

distcheck: dist
	@echo
	@echo -----------------------------------------------
	@echo $(PKG_EXE) ready for distribution
	@echo -----------------------------------------------

else

dist: $(PKG_TAR)

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
