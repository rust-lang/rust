# x86_64-pc-windows-gnu configuration
CROSS_PREFIX_x86_64-pc-windows-gnu=x86_64-w64-mingw32-
CC_x86_64-pc-windows-gnu=gcc
CXX_x86_64-pc-windows-gnu=g++
CPP_x86_64-pc-windows-gnu=gcc -E
AR_x86_64-pc-windows-gnu=ar
CFG_LIB_NAME_x86_64-pc-windows-gnu=$(1).dll
CFG_STATIC_LIB_NAME_x86_64-pc-windows-gnu=$(1).lib
CFG_LIB_GLOB_x86_64-pc-windows-gnu=$(1)-*.dll
CFG_LIB_DSYM_GLOB_x86_64-pc-windows-gnu=$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_x86_64-pc-windows-gnu := -m64 -D_WIN32_WINNT=0x0600 -D__USE_MINGW_ANSI_STDIO=1 $(CFLAGS)
CFG_GCCISH_CFLAGS_x86_64-pc-windows-gnu := -Wall -Werror -g -m64 -D_WIN32_WINNT=0x0600 -D__USE_MINGW_ANSI_STDIO=1 $(CFLAGS)
CFG_GCCISH_CXXFLAGS_x86_64-pc-windows-gnu := -fno-rtti $(CXXFLAGS)
CFG_GCCISH_LINK_FLAGS_x86_64-pc-windows-gnu := -shared -g -m64
CFG_GCCISH_DEF_FLAG_x86_64-pc-windows-gnu :=
CFG_LLC_FLAGS_x86_64-pc-windows-gnu :=
CFG_INSTALL_NAME_x86_64-pc-windows-gnu =
CFG_EXE_SUFFIX_x86_64-pc-windows-gnu := .exe
CFG_WINDOWSY_x86_64-pc-windows-gnu := 1
CFG_UNIXY_x86_64-pc-windows-gnu :=
CFG_LDPATH_x86_64-pc-windows-gnu :=
CFG_RUN_x86_64-pc-windows-gnu=$(2)
CFG_RUN_TARG_x86_64-pc-windows-gnu=$(call CFG_RUN_x86_64-pc-windows-gnu,,$(2))
CFG_GNU_TRIPLE_x86_64-pc-windows-gnu := x86_64-w64-mingw32
CFG_THIRD_PARTY_OBJECTS_x86_64-pc-windows-gnu := crt2.o dllcrt2.o
CFG_INSTALLED_OBJECTS_x86_64-pc-windows-gnu := crt2.o dllcrt2.o rsbegin.o rsend.o
