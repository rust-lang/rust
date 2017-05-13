# i586-pc-windows-msvc configuration
CC_i586-pc-windows-msvc=$(CFG_MSVC_CL_i386)
LINK_i586-pc-windows-msvc=$(CFG_MSVC_LINK_i386)
CXX_i586-pc-windows-msvc=$(CFG_MSVC_CL_i386)
CPP_i586-pc-windows-msvc=$(CFG_MSVC_CL_i386)
AR_i586-pc-windows-msvc=$(CFG_MSVC_LIB_i386)
CFG_LIB_NAME_i586-pc-windows-msvc=$(1).dll
CFG_STATIC_LIB_NAME_i586-pc-windows-msvc=$(1).lib
CFG_LIB_GLOB_i586-pc-windows-msvc=$(1)-*.{dll,lib}
CFG_LIB_DSYM_GLOB_i586-pc-windows-msvc=$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_i586-pc-windows-msvc :=
CFG_GCCISH_CFLAGS_i586-pc-windows-msvc := -MD -arch:IA32 -nologo
CFG_GCCISH_CXXFLAGS_i586-pc-windows-msvc := -MD -arch:IA32 -nologo
CFG_GCCISH_LINK_FLAGS_i586-pc-windows-msvc :=
CFG_GCCISH_DEF_FLAG_i586-pc-windows-msvc :=
CFG_LLC_FLAGS_i586-pc-windows-msvc :=
CFG_INSTALL_NAME_i586-pc-windows-msvc =
CFG_EXE_SUFFIX_i586-pc-windows-msvc := .exe
CFG_WINDOWSY_i586-pc-windows-msvc := 1
CFG_UNIXY_i586-pc-windows-msvc :=
CFG_LDPATH_i586-pc-windows-msvc :=
CFG_RUN_i586-pc-windows-msvc=$(2)
CFG_RUN_TARG_i586-pc-windows-msvc=$(call CFG_RUN_i586-pc-windows-msvc,,$(2))
CFG_GNU_TRIPLE_i586-pc-windows-msvc := i586-pc-win32

# Currently the build system is not configured to build jemalloc
# with MSVC, so we omit this optional dependency.
CFG_DISABLE_JEMALLOC_i586-pc-windows-msvc := 1
