# le32-unknown-nacl (portable, PNaCl)
ifneq ($(CFG_NACL_CROSS_PATH),)

CC_le32-unknown-nacl=$(shell $(CFG_PYTHON) $(CFG_NACL_CROSS_PATH)/tools/nacl_config.py -t pnacl --tool cc)
CXX_le32-unknown-nacl=$(shell $(CFG_PYTHON) $(CFG_NACL_CROSS_PATH)/tools/nacl_config.py -t pnacl --tool c++)
CPP_le32-unknown-nacl=$(CXX_le32-unknown-nacl) -E
AR_le32-unknown-nacl=$(shell $(CFG_PYTHON) $(CFG_NACL_CROSS_PATH)/tools/nacl_config.py -t pnacl --tool ar)

CFG_PNACL_TOOLCHAIN := $(abspath $(dir $(AR_le32-unknown-nacl)/../))

# Note: pso's aren't supported by PNaCl.
CFG_LIB_NAME_le32-unknown-nacl=lib$(1).pso
CFG_STATIC_LIB_NAME_le32-unknown-nacl=lib$(1).a
CFG_LIB_GLOB_le32-unknown-nacl=lib$(1)-*.pso
CFG_LIB_DSYM_GLOB_le32-unknown-nacl=lib$(1)-*.dylib.dSYM
CFG_GCCISH_CFLAGS_le32-unknown-nacl := -Wall -Wno-unused-variable -Wno-unused-value $(shell $(CFG_PYTHON) $(CFG_NACL_CROSS_PATH)/tools/nacl_config.py -t pnacl --cflags) -D_YUGA_LITTLE_ENDIAN=1 -D_YUGA_BIG_ENDIAN=0
CFG_GCCISH_CXXFLAGS_le32-unknown-nacl := -stdlib=libc++ $(CFG_GCCISH_CFLAGS_le32-unknown-nacl)
CFG_GCCISH_LINK_FLAGS_le32-unknown-nacl := -static -pthread -lm
CFG_GCCISH_DEF_FLAG_le32-unknown-nacl := -Wl,--export-dynamic,--dynamic-list=
CFG_GCCISH_PRE_LIB_FLAGS_le32-unknown-nacl := -Wl,-no-whole-archive
CFG_GCCISH_POST_LIB_FLAGS_le32-unknown-nacl :=
CFG_DEF_SUFFIX_le32-unknown-nacl := .le32.nacl.def
CFG_INSTALL_NAME_le32-unknown-nacl =
CFG_EXE_SUFFIX_le32-unknown-nacl = .pexe
CFG_WINDOWSY_le32-unknown-nacl :=
CFG_UNIXY_le32-unknown-nacl := 1
CFG_NACLY_le32-unknown-nacl := 1
CFG_PATH_MUNGE_le32-unknown-nacl := true
CFG_LDPATH_le32-unknown-nacl :=
CFG_RUN_le32-unknown-nacl=$(2)
CFG_RUN_TARG_le32-unknown-nacl=$(call CFG_RUN_le32-unknown-nacl,,$(2))
RUSTC_FLAGS_le32-unknown-nacl:=
RUSTC_CROSS_FLAGS_le32-unknown-nacl=-L $(CFG_NACL_CROSS_PATH)/lib/pnacl/Release -L $(CFG_PNACL_TOOLCHAIN)/lib/clang/3.7.0/lib/le32-nacl -L $(CFG_PNACL_TOOLCHAIN)/le32-nacl/usr/lib -L $(CFG_PNACL_TOOLCHAIN)/le32-nacl/lib
CFG_GNU_TRIPLE_le32-unknown-nacl := le32-unknown-nacl

# strdup isn't defined unless -std=gnu++11 is used :/
LLVM_FILTER_CXXFLAGS_le32-unknown-nacl := -std=c++11
LLVM_EXTRA_CXXFLAGS_le32-unknown-nacl := -std=gnu++11

endif
