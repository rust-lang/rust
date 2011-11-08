######################################################################
# Distribution
######################################################################


PKG_NAME := rust
PKG_VER  = $(shell date +"%Y-%m-%d")-snap
PKG_DIR = $(PKG_NAME)-$(PKG_VER)
PKG_TAR = $(PKG_DIR).tar.gz

ifdef CFG_MAKENSIS
PKG_NSI = $(S)src/etc/pkg/rust.nsi
PKG_EXE = $(PKG_DIR)-install.exe
endif

PKG_3RDPARTY := rt/valgrind.h rt/memcheck.h \
                rt/isaac/rand.h rt/isaac/standard.h \
                rt/uthash/uthash.h rt/uthash/utlist.h \
                rt/bigint/bigint.h rt/bigint/bigint_int.cpp \
                rt/bigint/bigint_ext.cpp rt/bigint/low_primes.h

PKG_UV := \
                $(wildcard $(S)src/libuv/*) \
                $(wildcard $(S)src/libuv/include/*) \
                $(wildcard $(S)src/libuv/include/*/*) \
                $(wildcard $(S)src/libuv/src/*) \
                $(wildcard $(S)src/libuv/src/*/*) \
                $(wildcard $(S)src/libuv/src/*/*/*)

PKG_PP_EXAMPLES = $(wildcard $(S)src/test/pretty/*.pp)

PKG_FILES = \
    $(wildcard $(S)src/etc/*.*)                \
    $(S)LICENSE.txt $(S)README                 \
    $(S)configure $(S)Makefile.in              \
    $(S)src/snapshots.txt                      \
    $(addprefix $(S)src/,                      \
      README comp/README                       \
      $(RUNTIME_CS) $(RUNTIME_HDR)             \
      $(RUNTIME_S)                             \
      rt/rustrt.def.in                         \
      rt/intrinsics/intrinsics.ll.in           \
      $(RUSTLLVM_LIB_CS) $(RUSTLLVM_OBJS_CS)   \
      $(RUSTLLVM_HDR)                          \
      rustllvm/rustllvm.def.in                 \
      $(PKG_3RDPARTY))                         \
    $(PKG_UV)                                  \
    $(COMPILER_INPUTS)                         \
    $(STDLIB_INPUTS)                           \
    $(ALL_TEST_INPUTS)                         \
    $(FUZZER_CRATE)                            \
    $(FUZZER_INPUTS)                           \
    $(COMPILETEST_CRATE)                       \
    $(COMPILETEST_INPUTS)                      \
    $(PKG_PP_EXAMPLES)                         \
    $(MKFILES)

dist: $(PKG_TAR) $(PKG_EXE)

nsis-dist: $(PKG_EXE)

lic.txt: $(S)LICENSE.txt
	@$(call E, crlf: $@)
	@$(Q)perl -pe 's@\r\n|\n@\r\n@go' <$< >$@

$(PKG_EXE): $(PKG_NSI) $(PKG_FILES) $(DOCS) $(SREQ3$(CFG_HOST_TRIPLE)) lic.txt
	@$(call E, makensis: $@)
	$(Q)makensis -NOCD -V1 "-XOutFile $@" "-XLicenseData lic.txt" $<
	$(Q)rm -f lic.txt

$(PKG_TAR): $(PKG_FILES)
	@$(call E, making dist dir)
	$(Q)rm -Rf dist
	$(Q)mkdir -p dist/$(PKG_DIR)
	$(Q)tar -c $(PKG_FILES) | tar -x -C dist/$(PKG_DIR)
	$(Q)tar -czf $(PKG_TAR) -C dist $(PKG_DIR)
	$(Q)rm -Rf dist

distcheck: $(PKG_TAR)
	$(Q)rm -Rf dist
	$(Q)mkdir -p dist
	@$(call E, unpacking $(PKG_TAR) in dist/$(PKG_DIR))
	$(Q)cd dist && tar -xzf ../$(PKG_TAR)
	@$(call E, configuring in dist/$(PKG_DIR)-build)
	$(Q)mkdir -p dist/$(PKG_DIR)-build
	$(Q)cd dist/$(PKG_DIR)-build && ../$(PKG_DIR)/configure
	@$(call E, making 'check' in dist/$(PKG_DIR)-build)
	$(Q)make -C dist/$(PKG_DIR)-build check
	@$(call E, making 'clean' in dist/$(PKG_DIR)-build)
	$(Q)make -C dist/$(PKG_DIR)-build clean
	$(Q)rm -Rf dist
	@echo
	@echo -----------------------------------------------
	@echo $(PKG_TAR) ready for distribution
	@echo -----------------------------------------------


