
CFG_GCCISH_CFLAGS := -fno-strict-aliasing
CFG_GCCISH_LINK_FLAGS :=

# On Darwin, we need to run dsymutil so the debugging information ends
# up in the right place.  On other platforms, it automatically gets
# embedded into the executable, so use a no-op command.
CFG_DSYMUTIL := true

ifeq ($(CFG_OSTYPE), FreeBSD)
  CFG_LIB_NAME=lib$(1).so
  CFG_GCCISH_CFLAGS += -fPIC -march=i686 -I/usr/local/include -O2
  CFG_GCCISH_LINK_FLAGS += -shared -fPIC -lpthread -lrt
  ifeq ($(CFG_CPUTYPE), x86_64)
    CFG_GCCISH_CFLAGS += -m32
    CFG_GCCISH_LINK_FLAGS += -m32
  endif
  CFG_UNIXY := 1
  CFG_LDENV := LD_LIBRARY_PATH
  CFG_DEF_SUFFIX := .bsd.def
endif

ifeq ($(CFG_OSTYPE), Linux)
  CFG_LIB_NAME=lib$(1).so
  CFG_GCCISH_CFLAGS += -fPIC -march=i686 -O2
  CFG_GCCISH_LINK_FLAGS += -shared -fPIC -ldl -lpthread -lrt
  CFG_GCCISH_DEF_FLAG := -Wl,--export-dynamic,--dynamic-list=
  CFG_GCCISH_PRE_LIB_FLAGS := -Wl,-whole-archive
  CFG_GCCISH_POST_LIB_FLAGS := -Wl,-no-whole-archive
  ifeq ($(CFG_CPUTYPE), x86_64)
    CFG_GCCISH_CFLAGS += -m32
    CFG_GCCISH_LINK_FLAGS += -m32
  endif
  CFG_UNIXY := 1
  CFG_LDENV := LD_LIBRARY_PATH
  CFG_DEF_SUFFIX := .linux.def
endif

ifeq ($(CFG_OSTYPE), Darwin)
  CFG_LIB_NAME=lib$(1).dylib
  CFG_UNIXY := 1
  CFG_LDENV := DYLD_LIBRARY_PATH
  CFG_GCCISH_LINK_FLAGS += -dynamiclib -lpthread
  CFG_GCCISH_DEF_FLAG := -Wl,-exported_symbols_list,
  # Darwin has a very blurry notion of "64 bit", and claims it's running
  # "on an i386" when the whole userspace is 64-bit and the compiler
  # emits 64-bit binaries by default. So we just force -m32 here. Smarter
  # approaches welcome!
  #
  # NB: Currently GCC's optimizer breaks rustrt (task-comm-1 hangs) on Darwin.
  CFG_GCC_CFLAGS += -m32 -O2
  CFG_CLANG_CFLAGS += -m32 -O2
  ifeq ($(CFG_CPUTYPE), x86_64)
    CFG_GCCISH_CFLAGS += -arch i386
    CFG_GCCISH_LINK_FLAGS += -arch i386
  endif
  CFG_GCCISH_LINK_FLAGS += -m32
  CFG_DSYMUTIL := dsymutil
  CFG_DEF_SUFFIX := .darwin.def
endif

ifneq ($(findstring MINGW,$(CFG_OSTYPE)),)
  CFG_WINDOWSY := 1
endif

CFG_TESTLIB=$(CFG_BUILD_DIR)/$(strip     \
 $(if $(findstring stage0,$(1)),         \
       stage0/lib,                           \
      $(if $(findstring stage1,$(1)),    \
           stage1/lib,                       \
          $(if $(findstring stage2,$(1)),\
               stage2/lib,                   \
               ))))

ifdef CFG_UNIXY
  CFG_INFO := $(info cfg: unix-y environment)

  CFG_PATH_MUNGE := true
  CFG_EXE_SUFFIX :=
  CFG_LDPATH :=$(CFG_LLVM_LIBDIR)
  CFG_RUN_TARG=$(CFG_LDENV)=$(CFG_BUILD_DIR)/$(1)/lib:$(CFG_LDPATH) $(2)
  CFG_RUN_TEST=\
      $(CFG_LDENV)=$(call CFG_TESTLIB,$(1)):$(CFG_LDPATH) \
      $(CFG_VALGRIND) $(1)

  ifdef CFG_ENABLE_MINGW_CROSS
    CFG_WINDOWSY := 1
    CFG_INFO := $(info cfg: mingw-cross)
    CFG_GCCISH_CROSS := i586-mingw32msvc-
    ifdef CFG_VALGRIND
      CFG_VALGRIND += wine
    endif

    CFG_GCCISH_CFLAGS := -fno-strict-aliasing -march=i586 -O2
    CFG_GCCISH_PRE_LIB_FLAGS :=
    CFG_GCCISH_POST_LIB_FLAGS :=
    CFG_GCCISH_DEF_FLAG :=
    CFG_GCCISH_LINK_FLAGS := -shared

    ifeq ($(CFG_CPUTYPE), x86_64)
      CFG_GCCISH_CFLAGS += -m32
      CFG_GCCISH_LINK_FLAGS += -m32
    endif
  endif
  ifdef CFG_VALGRIND
    CFG_VALGRIND += --leak-check=full \
                    --error-exitcode=1 \
                    --quiet --suppressions=$(CFG_SRC_DIR)src/etc/x86.supp
  endif
endif


ifdef CFG_WINDOWSY
  CFG_INFO := $(info cfg: windows-y environment)

  CFG_EXE_SUFFIX := .exe
  CFG_LIB_NAME=$(1).dll
  CFG_DEF_SUFFIX := .def
  CFG_LDPATH :=$(CFG_LLVM_BINDIR)
  CFG_LDPATH :=$(CFG_LDPATH):$$PATH
  CFG_RUN_TEST=PATH="$(CFG_LDPATH):$(call CFG_TESTLIB,$(1))" $(1)
  CFG_RUN_TARG=PATH="$(CFG_BUILD_DIR)/$(1)/lib:$(CFG_LDPATH)" $(2)

  ifndef CFG_ENABLE_MINGW_CROSS
    CFG_PATH_MUNGE := $(strip perl -i.bak -p             \
                             -e 's@\\(\S)@/\1@go;'       \
                             -e 's@^/([a-zA-Z])/@\1:/@o;')
    CFG_GCCISH_CFLAGS += -march=i686 -O2
    CFG_GCCISH_LINK_FLAGS += -shared -fPIC
  endif

endif


ifdef CFG_CLANG
  CFG_INFO := $(info cfg: using clang)
  CFG_GCCISH_CFLAGS += -Wall -Werror -fno-rtti -fno-exceptions -g
  CFG_GCCISH_LINK_FLAGS += -g
  CFG_COMPILE_C = $(CFG_GCCISH_CROSS)clang++ $(CFG_GCCISH_CFLAGS) \
    $(CFG_CLANG_CFLAGS) -c -o $(1) $(2)
  CFG_DEPEND_C = $(CFG_GCCISH_CROSS)clang++ $(CFG_GCCISH_CFLAGS) -MT "$(1)" \
    -MM $(2)
  CFG_LINK_C = $(CFG_GCCISH_CROSS)clang++ $(CFG_GCCISH_LINK_FLAGS) -o $(1) \
    $(CFG_GCCISH_DEF_FLAG)$(3) $(2)
else
ifdef CFG_GCC
  CFG_INFO := $(info cfg: using gcc)
  CFG_GCCISH_CFLAGS += -Wall -Werror -fno-rtti -fno-exceptions -g
  CFG_GCCISH_LINK_FLAGS += -g
  CFG_COMPILE_C = $(CFG_GCCISH_CROSS)g++ $(CFG_GCCISH_CFLAGS) \
    $(CFG_GCC_CFLAGS) -c -o $(1) $(2)
  CFG_DEPEND_C = $(CFG_GCCISH_CROSS)g++ $(CFG_GCCISH_CFLAGS) -MT "$(1)" \
    -MM $(2)
  CFG_LINK_C = $(CFG_GCCISH_CROSS)g++ $(CFG_GCCISH_LINK_FLAGS) -o $(1) \
               $(CFG_GCCISH_DEF_FLAG)$(3) $(2)
else
  CFG_ERR := $(error please try on a system with gcc or clang)
endif
endif
