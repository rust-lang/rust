# x86_64-pc-windows-msvc configuration
CC_x86_64-pc-windows-msvc="$(CFG_MSVC_CL)" -nologo
LINK_x86_64-pc-windows-msvc="$(CFG_MSVC_LINK)" -nologo
CXX_x86_64-pc-windows-msvc="$(CFG_MSVC_CL)" -nologo
CPP_x86_64-pc-windows-msvc="$(CFG_MSVC_CL)" -nologo
AR_x86_64-pc-windows-msvc="$(CFG_MSVC_LIB)" -nologo
CFG_LIB_NAME_x86_64-pc-windows-msvc=$(1).dll
CFG_STATIC_LIB_NAME_x86_64-pc-windows-msvc=$(1).lib
CFG_LIB_GLOB_x86_64-pc-windows-msvc=$(1)-*.dll
CFG_LIB_DSYM_GLOB_x86_64-pc-windows-msvc=$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_x86_64-pc-windows-msvc :=
CFG_GCCISH_CFLAGS_x86_64-pc-windows-msvc :=
CFG_GCCISH_CXXFLAGS_x86_64-pc-windows-msvc :=
CFG_GCCISH_LINK_FLAGS_x86_64-pc-windows-msvc :=
CFG_GCCISH_DEF_FLAG_x86_64-pc-windows-msvc :=
CFG_LLC_FLAGS_x86_64-pc-windows-msvc :=
CFG_INSTALL_NAME_x86_64-pc-windows-msvc =
CFG_EXE_SUFFIX_x86_64-pc-windows-msvc := .exe
CFG_WINDOWSY_x86_64-pc-windows-msvc := 1
CFG_UNIXY_x86_64-pc-windows-msvc :=
CFG_LDPATH_x86_64-pc-windows-msvc :=
CFG_RUN_x86_64-pc-windows-msvc=$(2)
CFG_RUN_TARG_x86_64-pc-windows-msvc=$(call CFG_RUN_x86_64-pc-windows-msvc,,$(2))
CFG_GNU_TRIPLE_x86_64-pc-windows-msvc := x86_64-pc-win32

# These two environment variables are scraped by the `./configure` script and
# are necessary for `cl.exe` to find standard headers (the INCLUDE variable) and
# for `link.exe` to find standard libraries (the LIB variable).
ifdef CFG_MSVC_INCLUDE_PATH
export INCLUDE := $(CFG_MSVC_INCLUDE_PATH)
endif
ifdef CFG_MSVC_LIB_PATH
export LIB := $(CFG_MSVC_LIB_PATH)
endif

# Unfortunately `link.exe` is also a program in `/usr/bin` on MinGW installs,
# but it's not the one that we want. As a result we make sure that our detected
# `link.exe` shows up in PATH first.
ifdef CFG_MSVC_LINK
export PATH := $(CFG_MSVC_ROOT)/VC/bin/amd64:$(PATH)
endif

# There are more comments about this available in the target specification for
# Windows MSVC in the compiler, but the gist of it is that we use `llvm-ar.exe`
# instead of `lib.exe` for assembling archives, so we need to inject this custom
# dependency here.
NATIVE_TOOL_DEPS_core_T_x86_64-pc-windows-msvc += llvm-ar.exe
