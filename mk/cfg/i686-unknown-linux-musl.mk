# i686-unknown-linux-musl configuration
CC_i686-unknown-linux-musl=$(CFG_MUSL_ROOT)/bin/musl-gcc
CXX_i686-unknown-linux-musl=$(CXX)
CPP_i686-unknown-linux-musl=$(CFG_MUSL_ROOT)/bin/musl-gcc -E
AR_i686-unknown-linux-musl=$(AR)
CFG_INSTALL_ONLY_RLIB_i686-unknown-linux-musl = 1
CFG_LIB_NAME_i686-unknown-linux-musl=lib$(1).so
CFG_STATIC_LIB_NAME_i686-unknown-linux-musl=lib$(1).a
CFG_LIB_GLOB_i686-unknown-linux-musl=lib$(1)-*.so
CFG_JEMALLOC_CFLAGS_i686-unknown-linux-musl := -m32 -Wl,-melf_i386 -Wa,-mrelax-relocations=no
CFG_GCCISH_CFLAGS_i686-unknown-linux-musl :=  -g -fPIC -m32 -Wl,-melf_i386 -Wa,-mrelax-relocations=no
CFG_GCCISH_CXXFLAGS_i686-unknown-linux-musl :=
CFG_GCCISH_LINK_FLAGS_i686-unknown-linux-musl :=
CFG_GCCISH_DEF_FLAG_i686-unknown-linux-musl :=
CFG_LLC_FLAGS_i686-unknown-linux-musl :=
CFG_INSTALL_NAME_i686-unknown-linux-musl =
CFG_EXE_SUFFIX_i686-unknown-linux-musl =
CFG_WINDOWSY_i686-unknown-linux-musl :=
CFG_UNIXY_i686-unknown-linux-musl := 1
CFG_LDPATH_i686-unknown-linux-musl :=
CFG_RUN_i686-unknown-linux-musl=$(2)
CFG_RUN_TARG_i686-unknown-linux-musl=$(call CFG_RUN_i686-unknown-linux-musl,,$(2))
CFG_GNU_TRIPLE_i686-unknown-linux-musl := i686-unknown-linux-musl
CFG_THIRD_PARTY_OBJECTS_i686-unknown-linux-musl := crt1.o crti.o crtn.o
CFG_INSTALLED_OBJECTS_i686-unknown-linux-musl := crt1.o crti.o crtn.o

NATIVE_DEPS_libc_T_i686-unknown-linux-musl += libc.a
NATIVE_DEPS_std_T_i686-unknown-linux-musl += crt1.o crti.o crtn.o
NATIVE_DEPS_unwind_T_i686-unknown-linux-musl += libunwind.a
