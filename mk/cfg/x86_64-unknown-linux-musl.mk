# x86_64-unknown-linux-musl configuration
CC_x86_64-unknown-linux-musl=$(CFG_MUSL_ROOT)/bin/musl-gcc
CXX_x86_64-unknown-linux-musl=$(CXX)
CPP_x86_64-unknown-linux-musl=$(CFG_MUSL_ROOT)/bin/musl-gcc -E
AR_x86_64-unknown-linux-musl=$(AR)
CFG_INSTALL_ONLY_RLIB_x86_64-unknown-linux-musl = 1
CFG_LIB_NAME_x86_64-unknown-linux-musl=lib$(1).so
CFG_STATIC_LIB_NAME_x86_64-unknown-linux-musl=lib$(1).a
CFG_LIB_GLOB_x86_64-unknown-linux-musl=lib$(1)-*.so
CFG_JEMALLOC_CFLAGS_x86_64-unknown-linux-musl := -m64 -Wa,-mrelax-relocations=no
CFG_GCCISH_CFLAGS_x86_64-unknown-linux-musl :=  -g -fPIC -m64 -Wa,-mrelax-relocations=no
CFG_GCCISH_CXXFLAGS_x86_64-unknown-linux-musl :=
CFG_GCCISH_LINK_FLAGS_x86_64-unknown-linux-musl :=
CFG_GCCISH_DEF_FLAG_x86_64-unknown-linux-musl :=
CFG_LLC_FLAGS_x86_64-unknown-linux-musl :=
CFG_INSTALL_NAME_x86_64-unknown-linux-musl =
CFG_EXE_SUFFIX_x86_64-unknown-linux-musl =
CFG_WINDOWSY_x86_64-unknown-linux-musl :=
CFG_UNIXY_x86_64-unknown-linux-musl := 1
CFG_LDPATH_x86_64-unknown-linux-musl :=
CFG_RUN_x86_64-unknown-linux-musl=$(2)
CFG_RUN_TARG_x86_64-unknown-linux-musl=$(call CFG_RUN_x86_64-unknown-linux-musl,,$(2))
CFG_GNU_TRIPLE_x86_64-unknown-linux-musl := x86_64-unknown-linux-musl
CFG_THIRD_PARTY_OBJECTS_x86_64-unknown-linux-musl := crt1.o crti.o crtn.o
CFG_INSTALLED_OBJECTS_x86_64-unknown-linux-musl := crt1.o crti.o crtn.o

NATIVE_DEPS_libc_T_x86_64-unknown-linux-musl += libc.a
NATIVE_DEPS_std_T_x86_64-unknown-linux-musl += crt1.o crti.o crtn.o
NATIVE_DEPS_unwind_T_x86_64-unknown-linux-musl += libunwind.a
