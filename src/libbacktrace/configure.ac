# configure.ac -- Backtrace configure script.
# Copyright (C) 2012-2016 Free Software Foundation, Inc.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

#     (1) Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer. 

#     (2) Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in
#     the documentation and/or other materials provided with the
#     distribution.  
    
#     (3) The name of the author may not be used to
#     endorse or promote products derived from this software without
#     specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

AC_PREREQ(2.64)
AC_INIT(package-unused, version-unused,, libbacktrace)
AC_CONFIG_SRCDIR(backtrace.h)
AC_CONFIG_HEADER(config.h)

if test -n "${with_target_subdir}"; then
  AM_ENABLE_MULTILIB(, ..)
fi

AC_CANONICAL_SYSTEM
target_alias=${target_alias-$host_alias}

AC_USE_SYSTEM_EXTENSIONS

libtool_VERSION=1:0:0
AC_SUBST(libtool_VERSION)

# 1.11.1: Require that version of automake.
# foreign: Don't require README, INSTALL, NEWS, etc.
# no-define: Don't define PACKAGE and VERSION.
# no-dependencies: Don't generate automatic dependencies.
#    (because it breaks when using bootstrap-lean, since some of the
#    headers are gone at "make install" time).
# -Wall: Issue all automake warnings.
# -Wno-portability: Don't warn about constructs supported by GNU make.
#    (because GCC requires GNU make anyhow).
AM_INIT_AUTOMAKE([1.11.1 foreign no-dist no-define no-dependencies -Wall -Wno-portability])

AM_MAINTAINER_MODE

AC_ARG_WITH(target-subdir,
[  --with-target-subdir=SUBDIR      Configuring in a subdirectory for target])

# We must force CC to /not/ be precious variables; otherwise
# the wrong, non-multilib-adjusted value will be used in multilibs.
# As a side effect, we have to subst CFLAGS ourselves.
m4_rename([_AC_ARG_VAR_PRECIOUS],[backtrace_PRECIOUS])
m4_define([_AC_ARG_VAR_PRECIOUS],[])
AC_PROG_CC
m4_rename_force([backtrace_PRECIOUS],[_AC_ARG_VAR_PRECIOUS])

AC_SUBST(CFLAGS)

AC_PROG_RANLIB

AC_PROG_AWK
case "$AWK" in
"") AC_MSG_ERROR([can't build without awk]) ;;
esac

LT_INIT
AM_PROG_LIBTOOL

backtrace_supported=yes

if test -n "${with_target_subdir}"; then
  # We are compiling a GCC library.  We can assume that the unwind
  # library exists.
  BACKTRACE_FILE="backtrace.lo simple.lo"
else
  AC_CHECK_HEADER([unwind.h],
    [AC_CHECK_FUNC([_Unwind_Backtrace],
	           [BACKTRACE_FILE="backtrace.lo simple.lo"],
		   [BACKTRACE_FILE="nounwind.lo"
                    backtrace_supported=no])],
    [BACKTRACE_FILE="nounwind.lo"
     backtrace_supported=no])
fi
AC_SUBST(BACKTRACE_FILE)

EXTRA_FLAGS=
if test -n "${with_target_subdir}"; then
  EXTRA_FLAGS="-funwind-tables -frandom-seed=\$@"
else
  AC_CACHE_CHECK([for -funwind-tables option],
    [libbacktrace_cv_c_unwind_tables],
    [CFLAGS_hold="$CFLAGS"
     CFLAGS="$CFLAGS -funwind-tables"
     AC_COMPILE_IFELSE(
       [AC_LANG_PROGRAM([static int f() { return 0; }], [return f();])],
       [libbacktrace_cv_c_unwind_tables=yes],
       [libbacktrace_cv_c_unwind_tables=no])
     CFLAGS="$CFLAGS_hold"])
  if test "$libbacktrace_cv_c_unwind_tables" = "yes"; then
    EXTRA_FLAGS=-funwind-tables
  fi
  AC_CACHE_CHECK([for -frandom-seed=string option],
    [libbacktrace_cv_c_random_seed_string],
    [CFLAGS_hold="$CFLAGS"
     CFLAGS="$CFLAGS -frandom-seed=conftest.lo"
     AC_COMPILE_IFELSE(
       [AC_LANG_PROGRAM([], [return 0;])],
       [libbacktrace_cv_c_random_seed_string=yes],
       [libbacktrace_cv_c_random_seed_string=no])
     CFLAGS="$CFLAGS_hold"])
  if test "$libbacktrace_cv_c_random_seed_string" = "yes"; then
    EXTRA_FLAGS="$EXTRA_FLAGS -frandom-seed=\$@"
  fi
fi
AC_SUBST(EXTRA_FLAGS)

ACX_PROG_CC_WARNING_OPTS([-W -Wall -Wwrite-strings -Wstrict-prototypes \
			  -Wmissing-prototypes -Wold-style-definition \
			  -Wmissing-format-attribute -Wcast-qual],
			  [WARN_FLAGS])

if test -n "${with_target_subdir}"; then
  WARN_FLAGS="$WARN_FLAGS -Werror"
fi

AC_SUBST(WARN_FLAGS)

if test -n "${with_target_subdir}"; then
  GCC_CHECK_UNWIND_GETIPINFO
else
  ac_save_CFFLAGS="$CFLAGS"
  CFLAGS="$CFLAGS -Werror-implicit-function-declaration"
  AC_MSG_CHECKING([for _Unwind_GetIPInfo])
  AC_LINK_IFELSE(
    [AC_LANG_PROGRAM(
       [#include "unwind.h"
	struct _Unwind_Context *context;
	int ip_before_insn = 0;],
	[return _Unwind_GetIPInfo (context, &ip_before_insn);])],
	[have_unwind_getipinfo=yes], [have_unwind_getipinfo=no])
  CFLAGS="$ac_save_CFLAGS"
  AC_MSG_RESULT([$have_unwind_getipinfo])
  if test "$have_unwind_getipinfo" = "yes"; then
    AC_DEFINE(HAVE_GETIPINFO, 1, [Define if _Unwind_GetIPInfo is available.])
  fi
fi

# Enable --enable-host-shared.
AC_ARG_ENABLE(host-shared,
[AS_HELP_STRING([--enable-host-shared],
		[build host code as shared libraries])],
[PIC_FLAG=-fPIC], [PIC_FLAG=])
AC_SUBST(PIC_FLAG)

# Test for __sync support.
AC_CACHE_CHECK([__sync extensions],
[libbacktrace_cv_sys_sync],
[if test -n "${with_target_subdir}"; then
   case "${host}" in
   hppa*-*-hpux*) libbacktrace_cv_sys_sync=no ;;
   *) libbacktrace_cv_sys_sync=yes ;;
   esac
 else
   AC_LINK_IFELSE(
     [AC_LANG_PROGRAM([int i;],
                      [__sync_bool_compare_and_swap (&i, i, i);
                       __sync_lock_test_and_set (&i, 1);
                       __sync_lock_release (&i);])],
     [libbacktrace_cv_sys_sync=yes],
     [libbacktrace_cv_sys_sync=no])
 fi])
BACKTRACE_SUPPORTS_THREADS=0
if test "$libbacktrace_cv_sys_sync" = "yes"; then
  BACKTRACE_SUPPORTS_THREADS=1
  AC_DEFINE([HAVE_SYNC_FUNCTIONS], 1,
	    [Define to 1 if you have the __sync functions])
fi
AC_SUBST(BACKTRACE_SUPPORTS_THREADS)

# Test for __atomic support.
AC_CACHE_CHECK([__atomic extensions],
[libbacktrace_cv_sys_atomic],
[if test -n "${with_target_subdir}"; then
   libbacktrace_cv_sys_atomic=yes
 else
   AC_LINK_IFELSE(
     [AC_LANG_PROGRAM([int i;],
     		      [__atomic_load_n (&i, __ATOMIC_ACQUIRE);
		       __atomic_store_n (&i, 1, __ATOMIC_RELEASE);])],
     [libbacktrace_cv_sys_atomic=yes],
     [libbacktrace_cv_sys_atomic=no])
 fi])
if test "$libbacktrace_cv_sys_atomic" = "yes"; then
  AC_DEFINE([HAVE_ATOMIC_FUNCTIONS], 1,
	    [Define to 1 if you have the __atomic functions])
fi

# The library needs to be able to read the executable itself.  Compile
# a file to determine the executable format.  The awk script
# filetype.awk prints out the file type.
AC_CACHE_CHECK([output filetype],
[libbacktrace_cv_sys_filetype],
[filetype=
AC_COMPILE_IFELSE(
  [AC_LANG_PROGRAM([int i;], [int j;])],
  [filetype=`${AWK} -f $srcdir/filetype.awk conftest.$ac_objext`],
  [AC_MSG_FAILURE([compiler failed])])
libbacktrace_cv_sys_filetype=$filetype])

# Match the file type to decide what files to compile.
FORMAT_FILE=
backtrace_supports_data=yes
case "$libbacktrace_cv_sys_filetype" in
elf*) FORMAT_FILE="elf.lo" ;;
pecoff) FORMAT_FILE="pecoff.lo"
        backtrace_supports_data=no
	;;
*) AC_MSG_WARN([could not determine output file type])
   FORMAT_FILE="unknown.lo"
   backtrace_supported=no
   ;;
esac
AC_SUBST(FORMAT_FILE)

# ELF defines.
elfsize=
case "$libbacktrace_cv_sys_filetype" in
elf32) elfsize=32 ;;
elf64) elfsize=64 ;;
*)     elfsize=unused
esac
AC_DEFINE_UNQUOTED([BACKTRACE_ELF_SIZE], [$elfsize], [ELF size: 32 or 64])

BACKTRACE_SUPPORTED=0
if test "$backtrace_supported" = "yes"; then
  BACKTRACE_SUPPORTED=1
fi
AC_SUBST(BACKTRACE_SUPPORTED)

BACKTRACE_SUPPORTS_DATA=0
if test "$backtrace_supports_data" = "yes"; then
  BACKTRACE_SUPPORTS_DATA=1
fi
AC_SUBST(BACKTRACE_SUPPORTS_DATA)

GCC_HEADER_STDINT(gstdint.h)

AC_CHECK_HEADERS(sys/mman.h)
if test "$ac_cv_header_sys_mman_h" = "no"; then
  have_mmap=no
else
  if test -n "${with_target_subdir}"; then
    # When built as a GCC target library, we can't do a link test.  We
    # simply assume that if we have mman.h, we have mmap.
    have_mmap=yes
    case "${host}" in
    spu-*-*|*-*-msdosdjgpp)
        # The SPU does not have mmap, but it has a sys/mman.h header file
        # containing "mmap_eaddr" and the mmap flags, confusing the test.
        # DJGPP also has sys/man.h, but no mmap
	have_mmap=no ;;
    esac
  else
    AC_CHECK_FUNC(mmap, [have_mmap=yes], [have_mmap=no])
  fi
fi
if test "$have_mmap" = "no"; then
  VIEW_FILE=read.lo
  ALLOC_FILE=alloc.lo
else
  VIEW_FILE=mmapio.lo
  AC_PREPROC_IFELSE([
#include <sys/mman.h>
#if !defined(MAP_ANONYMOUS) && !defined(MAP_ANON)
  #error no MAP_ANONYMOUS
#endif
], [ALLOC_FILE=mmap.lo], [ALLOC_FILE=alloc.lo])
fi
AC_SUBST(VIEW_FILE)
AC_SUBST(ALLOC_FILE)

BACKTRACE_USES_MALLOC=0
if test "$ALLOC_FILE" = "alloc.lo"; then
  BACKTRACE_USES_MALLOC=1
fi
AC_SUBST(BACKTRACE_USES_MALLOC)

# Check for dl_iterate_phdr.
AC_CHECK_HEADERS(link.h)
if test "$ac_cv_header_link_h" = "no"; then
  have_dl_iterate_phdr=no
else
  if test -n "${with_target_subdir}"; then
    # When built as a GCC target library, we can't do a link test.
    AC_EGREP_HEADER([dl_iterate_phdr], [link.h], [have_dl_iterate_phdr=yes],
		    [have_dl_iterate_phdr=no])
    case "${host}" in
    *-*-solaris2.10*)
        # Avoid dl_iterate_phdr on Solaris 10, where it is in the
    	# header file but is only in -ldl.
	have_dl_iterate_phdr=no ;;
    esac
  else
    AC_CHECK_FUNC([dl_iterate_phdr], [have_dl_iterate_phdr=yes],
		  [have_dl_iterate_phdr=no])
  fi
fi
if test "$have_dl_iterate_phdr" = "yes"; then
  AC_DEFINE(HAVE_DL_ITERATE_PHDR, 1, [Define if dl_iterate_phdr is available.])
fi

# Check for the fcntl function.
if test -n "${with_target_subdir}"; then
   case "${host}" in
   *-*-mingw*) have_fcntl=no ;;
   spu-*-*) have_fcntl=no ;;
   *) have_fcntl=yes ;;
   esac
else
  AC_CHECK_FUNC(fcntl, [have_fcntl=yes], [have_fcntl=no])
fi
if test "$have_fcntl" = "yes"; then
  AC_DEFINE([HAVE_FCNTL], 1,
	    [Define to 1 if you have the fcntl function])
fi

AC_CHECK_DECLS(strnlen)

# Check for getexecname function.
if test -n "${with_target_subdir}"; then
   case "${host}" in
   *-*-solaris2*) have_getexecname=yes ;;
   *) have_getexecname=no ;;
   esac
else
  AC_CHECK_FUNC(getexecname, [have_getexecname=yes], [have_getexecname=no])
fi
if test "$have_getexecname" = "yes"; then
  AC_DEFINE(HAVE_GETEXECNAME, 1, [Define if getexecname is available.])
fi

AC_CACHE_CHECK([whether tests can run],
  [libbacktrace_cv_sys_native],
  [AC_RUN_IFELSE([AC_LANG_PROGRAM([], [return 0;])],
     [libbacktrace_cv_sys_native=yes],
     [libbacktrace_cv_sys_native=no],
     [libbacktrace_cv_sys_native=no])])
AM_CONDITIONAL(NATIVE, test "$libbacktrace_cv_sys_native" = "yes")

if test "${multilib}" = "yes"; then
  multilib_arg="--enable-multilib"
else
  multilib_arg=
fi

AC_CONFIG_FILES(Makefile backtrace-supported.h)

# We need multilib support, but only if configuring for the target.
AC_CONFIG_COMMANDS([default],
[if test -n "$CONFIG_FILES"; then
   if test -n "${with_target_subdir}"; then
     # Multilibs need MULTISUBDIR defined correctly in certain makefiles so
     # that multilib installs will end up installed in the correct place.
     # The testsuite needs it for multilib-aware ABI baseline files.
     # To work around this not being passed down from config-ml.in ->
     # srcdir/Makefile.am -> srcdir/{src,libsupc++,...}/Makefile.am, manually
     # append it here.  Only modify Makefiles that have just been created.
     #
     # Also, get rid of this simulated-VPATH thing that automake does.
     cat > vpsed << \_EOF
  s!`test -f '$<' || echo '$(srcdir)/'`!!
_EOF
     for i in $SUBDIRS; do
      case $CONFIG_FILES in
       *${i}/Makefile*)
	 #echo "Adding MULTISUBDIR to $i/Makefile"
	 sed -f vpsed $i/Makefile > tmp
	 grep '^MULTISUBDIR =' Makefile >> tmp
	 mv tmp $i/Makefile
	 ;;
      esac
     done
     rm vpsed
   fi
 fi
],
[
# Variables needed in config.status (file generation) which aren't already
# passed by autoconf.
SUBDIRS="$SUBDIRS"
])

AC_OUTPUT
