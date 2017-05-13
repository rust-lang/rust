# i686-pc-windows-msvc configuration
CC_i686-pc-windows-msvc=$(CFG_MSVC_CL_i386)
LINK_i686-pc-windows-msvc=$(CFG_MSVC_LINK_i386)
CXX_i686-pc-windows-msvc=$(CFG_MSVC_CL_i386)
CPP_i686-pc-windows-msvc=$(CFG_MSVC_CL_i386)
AR_i686-pc-windows-msvc=$(CFG_MSVC_LIB_i386)
CFG_LIB_NAME_i686-pc-windows-msvc=$(1).dll
CFG_STATIC_LIB_NAME_i686-pc-windows-msvc=$(1).lib
CFG_LIB_GLOB_i686-pc-windows-msvc=$(1)-*.{dll,lib}
CFG_LIB_DSYM_GLOB_i686-pc-windows-msvc=$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_i686-pc-windows-msvc :=
CFG_GCCISH_CFLAGS_i686-pc-windows-msvc := -MD -nologo
CFG_GCCISH_CXXFLAGS_i686-pc-windows-msvc := -MD -nologo
CFG_GCCISH_LINK_FLAGS_i686-pc-windows-msvc :=
CFG_GCCISH_DEF_FLAG_i686-pc-windows-msvc :=
CFG_LLC_FLAGS_i686-pc-windows-msvc :=
CFG_INSTALL_NAME_i686-pc-windows-msvc =
CFG_EXE_SUFFIX_i686-pc-windows-msvc := .exe
CFG_WINDOWSY_i686-pc-windows-msvc := 1
CFG_UNIXY_i686-pc-windows-msvc :=
CFG_LDPATH_i686-pc-windows-msvc :=
CFG_RUN_i686-pc-windows-msvc=$(2)
CFG_RUN_TARG_i686-pc-windows-msvc=$(call CFG_RUN_i686-pc-windows-msvc,,$(2))
CFG_GNU_TRIPLE_i686-pc-windows-msvc := i686-pc-win32

# Currently the build system is not configured to build jemalloc
# with MSVC, so we omit this optional dependency.
CFG_DISABLE_JEMALLOC_i686-pc-windows-msvc := 1
