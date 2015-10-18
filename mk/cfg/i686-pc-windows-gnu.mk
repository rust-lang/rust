# i686-pc-windows-gnu configuration
CROSS_PREFIX_i686-pc-windows-gnu=i686-w64-mingw32-
CC_i686-pc-windows-gnu=gcc
CXX_i686-pc-windows-gnu=g++
CPP_i686-pc-windows-gnu=gcc -E
AR_i686-pc-windows-gnu=ar
CFG_LIB_NAME_i686-pc-windows-gnu=$(1).dll
CFG_STATIC_LIB_NAME_i686-pc-windows-gnu=$(1).lib
CFG_LIB_GLOB_i686-pc-windows-gnu=$(1)-*.dll
CFG_LIB_DSYM_GLOB_i686-pc-windows-gnu=$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_i686-pc-windows-gnu := -march=i686 -m32 -D_WIN32_WINNT=0x0600 -D__USE_MINGW_ANSI_STDIO=1 $(CFLAGS)
CFG_GCCISH_CFLAGS_i686-pc-windows-gnu := -Wall -Werror -g -m32 -D_WIN32_WINNT=0x0600 -D__USE_MINGW_ANSI_STDIO=1 $(CFLAGS)
CFG_GCCISH_CXXFLAGS_i686-pc-windows-gnu := -fno-rtti $(CXXFLAGS)
CFG_GCCISH_LINK_FLAGS_i686-pc-windows-gnu := -shared -g -m32
CFG_GCCISH_DEF_FLAG_i686-pc-windows-gnu :=
CFG_LLC_FLAGS_i686-pc-windows-gnu :=
CFG_INSTALL_NAME_i686-pc-windows-gnu =
CFG_EXE_SUFFIX_i686-pc-windows-gnu := .exe
CFG_WINDOWSY_i686-pc-windows-gnu := 1
CFG_UNIXY_i686-pc-windows-gnu :=
CFG_LDPATH_i686-pc-windows-gnu :=
CFG_RUN_i686-pc-windows-gnu=$(2)
CFG_RUN_TARG_i686-pc-windows-gnu=$(call CFG_RUN_i686-pc-windows-gnu,,$(2))
CFG_GNU_TRIPLE_i686-pc-windows-gnu := i686-w64-mingw32
CFG_LIBC_STARTUP_OBJECTS_i686-pc-windows-gnu := crt2.o dllcrt2.o
