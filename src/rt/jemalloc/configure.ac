dnl Process this file with autoconf to produce a configure script.
AC_INIT([Makefile.in])

dnl ============================================================================
dnl Custom macro definitions.

dnl JE_CFLAGS_APPEND(cflag)
AC_DEFUN([JE_CFLAGS_APPEND],
[
AC_MSG_CHECKING([whether compiler supports $1])
TCFLAGS="${CFLAGS}"
if test "x${CFLAGS}" = "x" ; then
  CFLAGS="$1"
else
  CFLAGS="${CFLAGS} $1"
fi
AC_COMPILE_IFELSE([AC_LANG_PROGRAM(
[[
]], [[
    return 0;
]])],
              AC_MSG_RESULT([yes]),
              AC_MSG_RESULT([no])
              [CFLAGS="${TCFLAGS}"]
)
])

dnl JE_COMPILABLE(label, hcode, mcode, rvar)
dnl 
dnl Use AC_LINK_IFELSE() rather than AC_COMPILE_IFELSE() so that linker errors
dnl cause failure.
AC_DEFUN([JE_COMPILABLE],
[
AC_CACHE_CHECK([whether $1 is compilable],
               [$4],
               [AC_LINK_IFELSE([AC_LANG_PROGRAM([$2],
                                                [$3])],
                               [$4=yes],
                               [$4=no])])
])

dnl ============================================================================

dnl Library revision.
rev=1
AC_SUBST([rev])

srcroot=$srcdir
if test "x${srcroot}" = "x." ; then
  srcroot=""
else
  srcroot="${srcroot}/"
fi
AC_SUBST([srcroot])
abs_srcroot="`cd \"${srcdir}\"; pwd`/"
AC_SUBST([abs_srcroot])

objroot=""
AC_SUBST([objroot])
abs_objroot="`pwd`/"
AC_SUBST([abs_objroot])

dnl Munge install path variables.
if test "x$prefix" = "xNONE" ; then
  prefix="/usr/local"
fi
if test "x$exec_prefix" = "xNONE" ; then
  exec_prefix=$prefix
fi
PREFIX=$prefix
AC_SUBST([PREFIX])
BINDIR=`eval echo $bindir`
BINDIR=`eval echo $BINDIR`
AC_SUBST([BINDIR])
INCLUDEDIR=`eval echo $includedir`
INCLUDEDIR=`eval echo $INCLUDEDIR`
AC_SUBST([INCLUDEDIR])
LIBDIR=`eval echo $libdir`
LIBDIR=`eval echo $LIBDIR`
AC_SUBST([LIBDIR])
DATADIR=`eval echo $datadir`
DATADIR=`eval echo $DATADIR`
AC_SUBST([DATADIR])
MANDIR=`eval echo $mandir`
MANDIR=`eval echo $MANDIR`
AC_SUBST([MANDIR])

dnl Support for building documentation.
AC_PATH_PROG([XSLTPROC], [xsltproc], [false], [$PATH])
if test -d "/usr/share/xml/docbook/stylesheet/docbook-xsl" ; then
  DEFAULT_XSLROOT="/usr/share/xml/docbook/stylesheet/docbook-xsl"
elif test -d "/usr/share/sgml/docbook/xsl-stylesheets" ; then
  DEFAULT_XSLROOT="/usr/share/sgml/docbook/xsl-stylesheets"
else
  dnl Documentation building will fail if this default gets used.
  DEFAULT_XSLROOT=""
fi
AC_ARG_WITH([xslroot],
  [AS_HELP_STRING([--with-xslroot=<path>], [XSL stylesheet root path])], [
if test "x$with_xslroot" = "xno" ; then
  XSLROOT="${DEFAULT_XSLROOT}"
else
  XSLROOT="${with_xslroot}"
fi
],
  XSLROOT="${DEFAULT_XSLROOT}"
)
AC_SUBST([XSLROOT])

dnl If CFLAGS isn't defined, set CFLAGS to something reasonable.  Otherwise,
dnl just prevent autoconf from molesting CFLAGS.
CFLAGS=$CFLAGS
AC_PROG_CC
if test "x$GCC" != "xyes" ; then
  AC_CACHE_CHECK([whether compiler is MSVC],
                 [je_cv_msvc],
                 [AC_COMPILE_IFELSE([AC_LANG_PROGRAM([],
                                                     [
#ifndef _MSC_VER
  int fail[-1];
#endif
])],
                               [je_cv_msvc=yes],
                               [je_cv_msvc=no])])
fi

if test "x$CFLAGS" = "x" ; then
  no_CFLAGS="yes"
  if test "x$GCC" = "xyes" ; then
    JE_CFLAGS_APPEND([-std=gnu99])
    JE_CFLAGS_APPEND([-Wall])
    JE_CFLAGS_APPEND([-pipe])
    JE_CFLAGS_APPEND([-g3])
  elif test "x$je_cv_msvc" = "xyes" ; then
    CC="$CC -nologo"
    JE_CFLAGS_APPEND([-Zi])
    JE_CFLAGS_APPEND([-MT])
    JE_CFLAGS_APPEND([-W3])
    CPPFLAGS="$CPPFLAGS -I${srcroot}/include/msvc_compat"
  fi
fi
dnl Append EXTRA_CFLAGS to CFLAGS, if defined.
if test "x$EXTRA_CFLAGS" != "x" ; then
  JE_CFLAGS_APPEND([$EXTRA_CFLAGS])
fi
AC_PROG_CPP

AC_CHECK_SIZEOF([void *])
if test "x${ac_cv_sizeof_void_p}" = "x8" ; then
  LG_SIZEOF_PTR=3
elif test "x${ac_cv_sizeof_void_p}" = "x4" ; then
  LG_SIZEOF_PTR=2
else
  AC_MSG_ERROR([Unsupported pointer size: ${ac_cv_sizeof_void_p}])
fi
AC_DEFINE_UNQUOTED([LG_SIZEOF_PTR], [$LG_SIZEOF_PTR])

AC_CHECK_SIZEOF([int])
if test "x${ac_cv_sizeof_int}" = "x8" ; then
  LG_SIZEOF_INT=3
elif test "x${ac_cv_sizeof_int}" = "x4" ; then
  LG_SIZEOF_INT=2
else
  AC_MSG_ERROR([Unsupported int size: ${ac_cv_sizeof_int}])
fi
AC_DEFINE_UNQUOTED([LG_SIZEOF_INT], [$LG_SIZEOF_INT])

AC_CHECK_SIZEOF([long])
if test "x${ac_cv_sizeof_long}" = "x8" ; then
  LG_SIZEOF_LONG=3
elif test "x${ac_cv_sizeof_long}" = "x4" ; then
  LG_SIZEOF_LONG=2
else
  AC_MSG_ERROR([Unsupported long size: ${ac_cv_sizeof_long}])
fi
AC_DEFINE_UNQUOTED([LG_SIZEOF_LONG], [$LG_SIZEOF_LONG])

AC_CHECK_SIZEOF([intmax_t])
if test "x${ac_cv_sizeof_intmax_t}" = "x16" ; then
  LG_SIZEOF_INTMAX_T=4
elif test "x${ac_cv_sizeof_intmax_t}" = "x8" ; then
  LG_SIZEOF_INTMAX_T=3
elif test "x${ac_cv_sizeof_intmax_t}" = "x4" ; then
  LG_SIZEOF_INTMAX_T=2
else
  AC_MSG_ERROR([Unsupported intmax_t size: ${ac_cv_sizeof_intmax_t}])
fi
AC_DEFINE_UNQUOTED([LG_SIZEOF_INTMAX_T], [$LG_SIZEOF_INTMAX_T])

AC_CANONICAL_HOST
dnl CPU-specific settings.
CPU_SPINWAIT=""
case "${host_cpu}" in
  i[[345]]86)
	;;
  i686)
	JE_COMPILABLE([__asm__], [], [[__asm__ volatile("pause"); return 0;]],
	              [je_cv_asm])
	if test "x${je_cv_asm}" = "xyes" ; then
	    CPU_SPINWAIT='__asm__ volatile("pause")'
	fi
	;;
  x86_64)
	JE_COMPILABLE([__asm__ syntax], [],
	              [[__asm__ volatile("pause"); return 0;]], [je_cv_asm])
	if test "x${je_cv_asm}" = "xyes" ; then
	    CPU_SPINWAIT='__asm__ volatile("pause")'
	fi
	;;
  *)
	;;
esac
AC_DEFINE_UNQUOTED([CPU_SPINWAIT], [$CPU_SPINWAIT])

LD_PRELOAD_VAR="LD_PRELOAD"
so="so"
importlib="${so}"
o="$ac_objext"
a="a"
exe="$ac_exeext"
libprefix="lib"
DSO_LDFLAGS='-shared -Wl,-soname,$(@F)'
RPATH='-Wl,-rpath,$(1)'
SOREV="${so}.${rev}"
PIC_CFLAGS='-fPIC -DPIC'
CTARGET='-o $@'
LDTARGET='-o $@'
EXTRA_LDFLAGS=
MKLIB='$(AR) crus $@'
CC_MM=1

dnl Platform-specific settings.  abi and RPATH can probably be determined
dnl programmatically, but doing so is error-prone, which makes it generally
dnl not worth the trouble.
dnl 
dnl Define cpp macros in CPPFLAGS, rather than doing AC_DEFINE(macro), since the
dnl definitions need to be seen before any headers are included, which is a pain
dnl to make happen otherwise.
default_munmap="1"
JEMALLOC_USABLE_SIZE_CONST="const"
case "${host}" in
  *-*-darwin*)
	CFLAGS="$CFLAGS"
	abi="macho"
	AC_DEFINE([JEMALLOC_PURGE_MADVISE_FREE], [ ])
	RPATH=""
	LD_PRELOAD_VAR="DYLD_INSERT_LIBRARIES"
	so="dylib"
	importlib="${so}"
	force_tls="0"
	DSO_LDFLAGS='-shared -Wl,-dylib_install_name,$(@F)'
	SOREV="${rev}.${so}"
	;;
  *-*-freebsd*)
	CFLAGS="$CFLAGS"
	abi="elf"
	AC_DEFINE([JEMALLOC_PURGE_MADVISE_FREE], [ ])
	force_lazy_lock="1"
	;;
  *-*-linux-android*)
	CFLAGS="$CFLAGS"
	CPPFLAGS="$CPPFLAGS -D_GNU_SOURCE"
	abi="elf"
	AC_DEFINE([JEMALLOC_HAS_ALLOCA_H])
	AC_DEFINE([JEMALLOC_PURGE_MADVISE_DONTNEED], [ ])
	AC_DEFINE([JEMALLOC_THREADED_INIT], [ ])
	default_munmap="0"
	;;
  *-*-linux*)
	CFLAGS="$CFLAGS"
	CPPFLAGS="$CPPFLAGS -D_GNU_SOURCE"
	abi="elf"
	AC_DEFINE([JEMALLOC_HAS_ALLOCA_H])
	AC_DEFINE([JEMALLOC_PURGE_MADVISE_DONTNEED], [ ])
	AC_DEFINE([JEMALLOC_THREADED_INIT], [ ])
	JEMALLOC_USABLE_SIZE_CONST=""
	default_munmap="0"
	;;
  *-*-netbsd*)
	AC_MSG_CHECKING([ABI])
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM(
[[#ifdef __ELF__
/* ELF */
#else
#error aout
#endif
]])],
                          [CFLAGS="$CFLAGS"; abi="elf"],
                          [abi="aout"])
	AC_MSG_RESULT([$abi])
	AC_DEFINE([JEMALLOC_PURGE_MADVISE_FREE], [ ])
	;;
  *-*-solaris2*)
	CFLAGS="$CFLAGS"
	abi="elf"
	RPATH='-Wl,-R,$(1)'
	dnl Solaris needs this for sigwait().
	CPPFLAGS="$CPPFLAGS -D_POSIX_PTHREAD_SEMANTICS"
	LIBS="$LIBS -lposix4 -lsocket -lnsl"
	;;
  *-ibm-aix*)
	if "$LG_SIZEOF_PTR" = "8"; then
	  dnl 64bit AIX
	  LD_PRELOAD_VAR="LDR_PRELOAD64"
	else
	  dnl 32bit AIX
	  LD_PRELOAD_VAR="LDR_PRELOAD"
	fi
	abi="xcoff"
	;;
  *-*-mingw*)
	abi="pecoff"
	force_tls="0"
	RPATH=""
	so="dll"
	if test "x$je_cv_msvc" = "xyes" ; then
	  importlib="lib"
	  DSO_LDFLAGS="-LD"
	  EXTRA_LDFLAGS="-link -DEBUG"
	  CTARGET='-Fo$@'
	  LDTARGET='-Fe$@'
	  MKLIB='lib -nologo -out:$@'
	  CC_MM=
        else
	  importlib="${so}"
	  DSO_LDFLAGS="-shared"
	fi
	a="lib"
	libprefix=""
	SOREV="${so}"
	PIC_CFLAGS=""
	;;
  *)
	AC_MSG_RESULT([Unsupported operating system: ${host}])
	abi="elf"
	;;
esac
AC_DEFINE_UNQUOTED([JEMALLOC_USABLE_SIZE_CONST], [$JEMALLOC_USABLE_SIZE_CONST])
AC_SUBST([abi])
AC_SUBST([RPATH])
AC_SUBST([LD_PRELOAD_VAR])
AC_SUBST([so])
AC_SUBST([importlib])
AC_SUBST([o])
AC_SUBST([a])
AC_SUBST([exe])
AC_SUBST([libprefix])
AC_SUBST([DSO_LDFLAGS])
AC_SUBST([EXTRA_LDFLAGS])
AC_SUBST([SOREV])
AC_SUBST([PIC_CFLAGS])
AC_SUBST([CTARGET])
AC_SUBST([LDTARGET])
AC_SUBST([MKLIB])
AC_SUBST([CC_MM])

if test "x$abi" != "xpecoff"; then
  dnl Heap profiling uses the log(3) function.
  LIBS="$LIBS -lm"
fi

JE_COMPILABLE([__attribute__ syntax],
              [static __attribute__((unused)) void foo(void){}],
              [],
              [je_cv_attribute])
if test "x${je_cv_attribute}" = "xyes" ; then
  AC_DEFINE([JEMALLOC_HAVE_ATTR], [ ])
  if test "x${GCC}" = "xyes" -a "x${abi}" = "xelf"; then
    JE_CFLAGS_APPEND([-fvisibility=hidden])
  fi
fi
dnl Check for tls_model attribute support (clang 3.0 still lacks support).
SAVED_CFLAGS="${CFLAGS}"
JE_CFLAGS_APPEND([-Werror])
JE_COMPILABLE([tls_model attribute], [],
              [static __thread int
               __attribute__((tls_model("initial-exec"))) foo;
               foo = 0;],
              [je_cv_tls_model])
CFLAGS="${SAVED_CFLAGS}"
if test "x${je_cv_tls_model}" = "xyes" ; then
  AC_DEFINE([JEMALLOC_TLS_MODEL],
            [__attribute__((tls_model("initial-exec")))])
else
  AC_DEFINE([JEMALLOC_TLS_MODEL], [ ])
fi

dnl Support optional additions to rpath.
AC_ARG_WITH([rpath],
  [AS_HELP_STRING([--with-rpath=<rpath>], [Colon-separated rpath (ELF systems only)])],
if test "x$with_rpath" = "xno" ; then
  RPATH_EXTRA=
else
  RPATH_EXTRA="`echo $with_rpath | tr \":\" \" \"`"
fi,
  RPATH_EXTRA=
)
AC_SUBST([RPATH_EXTRA])

dnl Disable rules that do automatic regeneration of configure output by default.
AC_ARG_ENABLE([autogen],
  [AS_HELP_STRING([--enable-autogen], [Automatically regenerate configure output])],
if test "x$enable_autogen" = "xno" ; then
  enable_autogen="0"
else
  enable_autogen="1"
fi
,
enable_autogen="0"
)
AC_SUBST([enable_autogen])

AC_PROG_INSTALL
AC_PROG_RANLIB
AC_PATH_PROG([AR], [ar], [false], [$PATH])
AC_PATH_PROG([LD], [ld], [false], [$PATH])
AC_PATH_PROG([AUTOCONF], [autoconf], [false], [$PATH])

public_syms="malloc_conf malloc_message malloc calloc posix_memalign aligned_alloc realloc free malloc_usable_size malloc_stats_print mallctl mallctlnametomib mallctlbymib"

dnl Check for allocator-related functions that should be wrapped.
AC_CHECK_FUNC([memalign],
	      [AC_DEFINE([JEMALLOC_OVERRIDE_MEMALIGN], [ ])
	       public_syms="${public_syms} memalign"])
AC_CHECK_FUNC([valloc],
	      [AC_DEFINE([JEMALLOC_OVERRIDE_VALLOC], [ ])
	       public_syms="${public_syms} valloc"])

dnl Support the experimental API by default.
AC_ARG_ENABLE([experimental],
  [AS_HELP_STRING([--disable-experimental],
   [Disable support for the experimental API])],
[if test "x$enable_experimental" = "xno" ; then
  enable_experimental="0"
else
  enable_experimental="1"
fi
],
[enable_experimental="1"]
)
if test "x$enable_experimental" = "x1" ; then
  AC_DEFINE([JEMALLOC_EXPERIMENTAL], [ ])
  public_syms="${public_syms} allocm dallocm nallocm rallocm sallocm"
fi
AC_SUBST([enable_experimental])

dnl Perform no name mangling by default.
AC_ARG_WITH([mangling],
  [AS_HELP_STRING([--with-mangling=<map>], [Mangle symbols in <map>])],
  [mangling_map="$with_mangling"], [mangling_map=""])
for nm in `echo ${mangling_map} |tr ',' ' '` ; do
  k="`echo ${nm} |tr ':' ' ' |awk '{print $1}'`"
  n="je_${k}"
  m=`echo ${nm} |tr ':' ' ' |awk '{print $2}'`
  AC_DEFINE_UNQUOTED([${n}], [${m}])
  dnl Remove key from public_syms so that it isn't redefined later.
  public_syms=`for sym in ${public_syms}; do echo "${sym}"; done |grep -v "^${k}\$" |tr '\n' ' '`
done

dnl Do not prefix public APIs by default.
AC_ARG_WITH([jemalloc_prefix],
  [AS_HELP_STRING([--with-jemalloc-prefix=<prefix>], [Prefix to prepend to all public APIs])],
  [JEMALLOC_PREFIX="$with_jemalloc_prefix"],
  [if test "x$abi" != "xmacho" -a "x$abi" != "xpecoff"; then
  JEMALLOC_PREFIX=""
else
  JEMALLOC_PREFIX="je_"
fi]
)
if test "x$JEMALLOC_PREFIX" != "x" ; then
  JEMALLOC_CPREFIX=`echo ${JEMALLOC_PREFIX} | tr "a-z" "A-Z"`
  AC_DEFINE_UNQUOTED([JEMALLOC_PREFIX], ["$JEMALLOC_PREFIX"])
  AC_DEFINE_UNQUOTED([JEMALLOC_CPREFIX], ["$JEMALLOC_CPREFIX"])
fi
dnl Generate macros to rename public symbols.  All public symbols are prefixed
dnl with je_ in the source code, so these macro definitions are needed even if
dnl --with-jemalloc-prefix wasn't specified.
for stem in ${public_syms}; do
  n="je_${stem}"
  m="${JEMALLOC_PREFIX}${stem}"
  AC_DEFINE_UNQUOTED([${n}], [${m}])
done

AC_ARG_WITH([export],
  [AS_HELP_STRING([--without-export], [disable exporting jemalloc public APIs])],
  [if test "x$with_export" = "xno"; then
  AC_DEFINE([JEMALLOC_EXPORT],[])
fi]
)

dnl Do not mangle library-private APIs by default.
AC_ARG_WITH([private_namespace],
  [AS_HELP_STRING([--with-private-namespace=<prefix>], [Prefix to prepend to all library-private APIs])],
  [JEMALLOC_PRIVATE_NAMESPACE="$with_private_namespace"],
  [JEMALLOC_PRIVATE_NAMESPACE=""]
)
AC_DEFINE_UNQUOTED([JEMALLOC_PRIVATE_NAMESPACE], ["$JEMALLOC_PRIVATE_NAMESPACE"])
if test "x$JEMALLOC_PRIVATE_NAMESPACE" != "x" ; then
  AC_DEFINE_UNQUOTED([JEMALLOC_N(string_that_no_one_should_want_to_use_as_a_jemalloc_private_namespace_prefix)], [${JEMALLOC_PRIVATE_NAMESPACE}##string_that_no_one_should_want_to_use_as_a_jemalloc_private_namespace_prefix])
else
  AC_DEFINE_UNQUOTED([JEMALLOC_N(string_that_no_one_should_want_to_use_as_a_jemalloc_private_namespace_prefix)], [string_that_no_one_should_want_to_use_as_a_jemalloc_private_namespace_prefix])
fi

dnl Do not add suffix to installed files by default.
AC_ARG_WITH([install_suffix],
  [AS_HELP_STRING([--with-install-suffix=<suffix>], [Suffix to append to all installed files])],
  [INSTALL_SUFFIX="$with_install_suffix"],
  [INSTALL_SUFFIX=]
)
install_suffix="$INSTALL_SUFFIX"
AC_SUBST([install_suffix])

cfgoutputs_in="${srcroot}Makefile.in"
cfgoutputs_in="${cfgoutputs_in} ${srcroot}doc/html.xsl.in"
cfgoutputs_in="${cfgoutputs_in} ${srcroot}doc/manpages.xsl.in"
cfgoutputs_in="${cfgoutputs_in} ${srcroot}doc/jemalloc.xml.in"
cfgoutputs_in="${cfgoutputs_in} ${srcroot}include/jemalloc/jemalloc.h.in"
cfgoutputs_in="${cfgoutputs_in} ${srcroot}include/jemalloc/internal/jemalloc_internal.h.in"
cfgoutputs_in="${cfgoutputs_in} ${srcroot}test/jemalloc_test.h.in"

cfgoutputs_out="Makefile"
cfgoutputs_out="${cfgoutputs_out} doc/html.xsl"
cfgoutputs_out="${cfgoutputs_out} doc/manpages.xsl"
cfgoutputs_out="${cfgoutputs_out} doc/jemalloc${install_suffix}.xml"
cfgoutputs_out="${cfgoutputs_out} include/jemalloc/jemalloc${install_suffix}.h"
cfgoutputs_out="${cfgoutputs_out} include/jemalloc/internal/jemalloc_internal.h"
cfgoutputs_out="${cfgoutputs_out} test/jemalloc_test.h"

cfgoutputs_tup="Makefile"
cfgoutputs_tup="${cfgoutputs_tup} doc/html.xsl:doc/html.xsl.in"
cfgoutputs_tup="${cfgoutputs_tup} doc/manpages.xsl:doc/manpages.xsl.in"
cfgoutputs_tup="${cfgoutputs_tup} doc/jemalloc${install_suffix}.xml:doc/jemalloc.xml.in"
cfgoutputs_tup="${cfgoutputs_tup} include/jemalloc/jemalloc${install_suffix}.h:include/jemalloc/jemalloc.h.in"
cfgoutputs_tup="${cfgoutputs_tup} include/jemalloc/internal/jemalloc_internal.h"
cfgoutputs_tup="${cfgoutputs_tup} test/jemalloc_test.h:test/jemalloc_test.h.in"

cfghdrs_in="${srcroot}include/jemalloc/jemalloc_defs.h.in"
cfghdrs_in="${cfghdrs_in} ${srcroot}include/jemalloc/internal/size_classes.sh"

cfghdrs_out="include/jemalloc/jemalloc_defs${install_suffix}.h"
cfghdrs_out="${cfghdrs_out} include/jemalloc/internal/size_classes.h"

cfghdrs_tup="include/jemalloc/jemalloc_defs${install_suffix}.h:include/jemalloc/jemalloc_defs.h.in"

dnl Do not silence irrelevant compiler warnings by default, since enabling this
dnl option incurs a performance penalty.
AC_ARG_ENABLE([cc-silence],
  [AS_HELP_STRING([--enable-cc-silence],
                  [Silence irrelevant compiler warnings])],
[if test "x$enable_cc_silence" = "xno" ; then
  enable_cc_silence="0"
else
  enable_cc_silence="1"
fi
],
[enable_cc_silence="0"]
)
if test "x$enable_cc_silence" = "x1" ; then
  AC_DEFINE([JEMALLOC_CC_SILENCE], [ ])
fi

dnl Do not compile with debugging by default.
AC_ARG_ENABLE([debug],
  [AS_HELP_STRING([--enable-debug], [Build debugging code (implies --enable-ivsalloc)])],
[if test "x$enable_debug" = "xno" ; then
  enable_debug="0"
else
  enable_debug="1"
fi
],
[enable_debug="0"]
)
if test "x$enable_debug" = "x1" ; then
  AC_DEFINE([JEMALLOC_DEBUG], [ ])
  enable_ivsalloc="1"
fi
AC_SUBST([enable_debug])

dnl Do not validate pointers by default.
AC_ARG_ENABLE([ivsalloc],
  [AS_HELP_STRING([--enable-ivsalloc], [Validate pointers passed through the public API])],
[if test "x$enable_ivsalloc" = "xno" ; then
  enable_ivsalloc="0"
else
  enable_ivsalloc="1"
fi
],
[enable_ivsalloc="0"]
)
if test "x$enable_ivsalloc" = "x1" ; then
  AC_DEFINE([JEMALLOC_IVSALLOC], [ ])
fi

dnl Only optimize if not debugging.
if test "x$enable_debug" = "x0" -a "x$no_CFLAGS" = "xyes" ; then
  dnl Make sure that an optimization flag was not specified in EXTRA_CFLAGS.
  optimize="no"
  echo "$EXTRA_CFLAGS" | grep "\-O" >/dev/null || optimize="yes"
  if test "x${optimize}" = "xyes" ; then
    if test "x$GCC" = "xyes" ; then
      JE_CFLAGS_APPEND([-O3])
      JE_CFLAGS_APPEND([-funroll-loops])
    elif test "x$je_cv_msvc" = "xyes" ; then
      JE_CFLAGS_APPEND([-O2])
    else
      JE_CFLAGS_APPEND([-O])
    fi
  fi
fi

dnl Enable statistics calculation by default.
AC_ARG_ENABLE([stats],
  [AS_HELP_STRING([--disable-stats],
                  [Disable statistics calculation/reporting])],
[if test "x$enable_stats" = "xno" ; then
  enable_stats="0"
else
  enable_stats="1"
fi
],
[enable_stats="1"]
)
if test "x$enable_stats" = "x1" ; then
  AC_DEFINE([JEMALLOC_STATS], [ ])
fi
AC_SUBST([enable_stats])

dnl Do not enable profiling by default.
AC_ARG_ENABLE([prof],
  [AS_HELP_STRING([--enable-prof], [Enable allocation profiling])],
[if test "x$enable_prof" = "xno" ; then
  enable_prof="0"
else
  enable_prof="1"
fi
],
[enable_prof="0"]
)
if test "x$enable_prof" = "x1" ; then
  backtrace_method=""
else
  backtrace_method="N/A"
fi

AC_ARG_ENABLE([prof-libunwind],
  [AS_HELP_STRING([--enable-prof-libunwind], [Use libunwind for backtracing])],
[if test "x$enable_prof_libunwind" = "xno" ; then
  enable_prof_libunwind="0"
else
  enable_prof_libunwind="1"
fi
],
[enable_prof_libunwind="0"]
)
AC_ARG_WITH([static_libunwind],
  [AS_HELP_STRING([--with-static-libunwind=<libunwind.a>],
  [Path to static libunwind library; use rather than dynamically linking])],
if test "x$with_static_libunwind" = "xno" ; then
  LUNWIND="-lunwind"
else
  if test ! -f "$with_static_libunwind" ; then
    AC_MSG_ERROR([Static libunwind not found: $with_static_libunwind])
  fi
  LUNWIND="$with_static_libunwind"
fi,
  LUNWIND="-lunwind"
)
if test "x$backtrace_method" = "x" -a "x$enable_prof_libunwind" = "x1" ; then
  AC_CHECK_HEADERS([libunwind.h], , [enable_prof_libunwind="0"])
  if test "x$LUNWIND" = "x-lunwind" ; then
    AC_CHECK_LIB([unwind], [backtrace], [LIBS="$LIBS $LUNWIND"],
                 [enable_prof_libunwind="0"])
  else
    LIBS="$LIBS $LUNWIND"
  fi
  if test "x${enable_prof_libunwind}" = "x1" ; then
    backtrace_method="libunwind"
    AC_DEFINE([JEMALLOC_PROF_LIBUNWIND], [ ])
  fi
fi

AC_ARG_ENABLE([prof-libgcc],
  [AS_HELP_STRING([--disable-prof-libgcc],
  [Do not use libgcc for backtracing])],
[if test "x$enable_prof_libgcc" = "xno" ; then
  enable_prof_libgcc="0"
else
  enable_prof_libgcc="1"
fi
],
[enable_prof_libgcc="1"]
)
if test "x$backtrace_method" = "x" -a "x$enable_prof_libgcc" = "x1" \
     -a "x$GCC" = "xyes" ; then
  AC_CHECK_HEADERS([unwind.h], , [enable_prof_libgcc="0"])
  AC_CHECK_LIB([gcc], [_Unwind_Backtrace], [LIBS="$LIBS -lgcc"], [enable_prof_libgcc="0"])
  dnl The following is conservative, in that it only has entries for CPUs on
  dnl which jemalloc has been tested.
  AC_MSG_CHECKING([libgcc-based backtracing reliability on ${host_cpu}])
  case "${host_cpu}" in
    i[[3456]]86)
      AC_MSG_RESULT([unreliable])
      enable_prof_libgcc="0";
      ;;
    x86_64)
      AC_MSG_RESULT([reliable])
      ;;
    *)
      AC_MSG_RESULT([unreliable])
      enable_prof_libgcc="0";
      ;;
  esac
  if test "x${enable_prof_libgcc}" = "x1" ; then
    backtrace_method="libgcc"
    AC_DEFINE([JEMALLOC_PROF_LIBGCC], [ ])
  fi
else
  enable_prof_libgcc="0"
fi

AC_ARG_ENABLE([prof-gcc],
  [AS_HELP_STRING([--disable-prof-gcc],
  [Do not use gcc intrinsics for backtracing])],
[if test "x$enable_prof_gcc" = "xno" ; then
  enable_prof_gcc="0"
else
  enable_prof_gcc="1"
fi
],
[enable_prof_gcc="1"]
)
if test "x$backtrace_method" = "x" -a "x$enable_prof_gcc" = "x1" \
     -a "x$GCC" = "xyes" ; then
  backtrace_method="gcc intrinsics"
  AC_DEFINE([JEMALLOC_PROF_GCC], [ ])
else
  enable_prof_gcc="0"
fi

if test "x$backtrace_method" = "x" ; then
  backtrace_method="none (disabling profiling)"
  enable_prof="0"
fi
AC_MSG_CHECKING([configured backtracing method])
AC_MSG_RESULT([$backtrace_method])
if test "x$enable_prof" = "x1" ; then
  if test "x${force_tls}" = "x0" ; then
    AC_MSG_ERROR([Heap profiling requires TLS]);
  fi
  force_tls="1"
  AC_DEFINE([JEMALLOC_PROF], [ ])
fi
AC_SUBST([enable_prof])

dnl Enable thread-specific caching by default.
AC_ARG_ENABLE([tcache],
  [AS_HELP_STRING([--disable-tcache], [Disable per thread caches])],
[if test "x$enable_tcache" = "xno" ; then
  enable_tcache="0"
else
  enable_tcache="1"
fi
],
[enable_tcache="1"]
)
if test "x$enable_tcache" = "x1" ; then
  AC_DEFINE([JEMALLOC_TCACHE], [ ])
fi
AC_SUBST([enable_tcache])

dnl Disable mremap() for huge realloc() by default.
AC_ARG_ENABLE([mremap],
  [AS_HELP_STRING([--enable-mremap], [Enable mremap(2) for huge realloc()])],
[if test "x$enable_mremap" = "xno" ; then
  enable_mremap="0"
else
  enable_mremap="1"
fi
],
[enable_mremap="0"]
)
if test "x$enable_mremap" = "x1" ; then
  JE_COMPILABLE([mremap(...MREMAP_FIXED...)], [
#define _GNU_SOURCE
#include <sys/mman.h>
], [
void *p = mremap((void *)0, 0, 0, MREMAP_MAYMOVE|MREMAP_FIXED, (void *)0);
], [je_cv_mremap_fixed])
  if test "x${je_cv_mremap_fixed}" = "xno" ; then
    enable_mremap="0"
  fi
fi
if test "x$enable_mremap" = "x1" ; then
  AC_DEFINE([JEMALLOC_MREMAP], [ ])
fi
AC_SUBST([enable_mremap])

dnl Enable VM deallocation via munmap() by default.
AC_ARG_ENABLE([munmap],
  [AS_HELP_STRING([--disable-munmap], [Disable VM deallocation via munmap(2)])],
[if test "x$enable_munmap" = "xno" ; then
  enable_munmap="0"
else
  enable_munmap="1"
fi
],
[enable_munmap="${default_munmap}"]
)
if test "x$enable_munmap" = "x1" ; then
  AC_DEFINE([JEMALLOC_MUNMAP], [ ])
fi
AC_SUBST([enable_munmap])

dnl Do not enable allocation from DSS by default.
AC_ARG_ENABLE([dss],
  [AS_HELP_STRING([--enable-dss], [Enable allocation from DSS])],
[if test "x$enable_dss" = "xno" ; then
  enable_dss="0"
else
  enable_dss="1"
fi
],
[enable_dss="0"]
)
dnl Check whether the BSD/SUSv1 sbrk() exists.  If not, disable DSS support.
AC_CHECK_FUNC([sbrk], [have_sbrk="1"], [have_sbrk="0"])
if test "x$have_sbrk" = "x1" ; then
  AC_DEFINE([JEMALLOC_HAVE_SBRK], [ ])
else
  enable_dss="0"
fi

if test "x$enable_dss" = "x1" ; then
  AC_DEFINE([JEMALLOC_DSS], [ ])
fi
AC_SUBST([enable_dss])

dnl Support the junk/zero filling option by default.
AC_ARG_ENABLE([fill],
  [AS_HELP_STRING([--disable-fill],
                  [Disable support for junk/zero filling, quarantine, and redzones])],
[if test "x$enable_fill" = "xno" ; then
  enable_fill="0"
else
  enable_fill="1"
fi
],
[enable_fill="1"]
)
if test "x$enable_fill" = "x1" ; then
  AC_DEFINE([JEMALLOC_FILL], [ ])
fi
AC_SUBST([enable_fill])

dnl Disable utrace(2)-based tracing by default.
AC_ARG_ENABLE([utrace],
  [AS_HELP_STRING([--enable-utrace], [Enable utrace(2)-based tracing])],
[if test "x$enable_utrace" = "xno" ; then
  enable_utrace="0"
else
  enable_utrace="1"
fi
],
[enable_utrace="0"]
)
JE_COMPILABLE([utrace(2)], [
#include <sys/types.h>
#include <sys/param.h>
#include <sys/time.h>
#include <sys/uio.h>
#include <sys/ktrace.h>
], [
	utrace((void *)0, 0);
], [je_cv_utrace])
if test "x${je_cv_utrace}" = "xno" ; then
  enable_utrace="0"
fi
if test "x$enable_utrace" = "x1" ; then
  AC_DEFINE([JEMALLOC_UTRACE], [ ])
fi
AC_SUBST([enable_utrace])

dnl Support Valgrind by default.
AC_ARG_ENABLE([valgrind],
  [AS_HELP_STRING([--disable-valgrind], [Disable support for Valgrind])],
[if test "x$enable_valgrind" = "xno" ; then
  enable_valgrind="0"
else
  enable_valgrind="1"
fi
],
[enable_valgrind="1"]
)
if test "x$enable_valgrind" = "x1" ; then
  JE_COMPILABLE([valgrind], [
#include <valgrind/valgrind.h>
#include <valgrind/memcheck.h>

#if !defined(VALGRIND_RESIZEINPLACE_BLOCK)
#  error "Incompatible Valgrind version"
#endif
], [], [je_cv_valgrind])
  if test "x${je_cv_valgrind}" = "xno" ; then
    enable_valgrind="0"
  fi
  if test "x$enable_valgrind" = "x1" ; then
    AC_DEFINE([JEMALLOC_VALGRIND], [ ])
  fi
fi
AC_SUBST([enable_valgrind])

dnl Do not support the xmalloc option by default.
AC_ARG_ENABLE([xmalloc],
  [AS_HELP_STRING([--enable-xmalloc], [Support xmalloc option])],
[if test "x$enable_xmalloc" = "xno" ; then
  enable_xmalloc="0"
else
  enable_xmalloc="1"
fi
],
[enable_xmalloc="0"]
)
if test "x$enable_xmalloc" = "x1" ; then
  AC_DEFINE([JEMALLOC_XMALLOC], [ ])
fi
AC_SUBST([enable_xmalloc])

AC_CACHE_CHECK([STATIC_PAGE_SHIFT],
               [je_cv_static_page_shift],
               AC_RUN_IFELSE([AC_LANG_PROGRAM(
[[
#include <strings.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <stdio.h>
]],
[[
    int result;
    FILE *f;

#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    result = si.dwPageSize;
#else
    result = sysconf(_SC_PAGESIZE);
#endif
    if (result == -1) {
	return 1;
    }
    result = ffsl(result) - 1;

    f = fopen("conftest.out", "w");
    if (f == NULL) {
	return 1;
    }
    fprintf(f, "%d\n", result);
    fclose(f);

    return 0;
]])],
                             [je_cv_static_page_shift=`cat conftest.out`],
                             [je_cv_static_page_shift=undefined],
                             [je_cv_static_page_shift=12]))

if test "x$je_cv_static_page_shift" != "xundefined"; then
   AC_DEFINE_UNQUOTED([STATIC_PAGE_SHIFT], [$je_cv_static_page_shift])
else
   AC_MSG_ERROR([cannot determine value for STATIC_PAGE_SHIFT])
fi

dnl ============================================================================
dnl jemalloc configuration.
dnl 

dnl Set VERSION if source directory has an embedded git repository.
if test -d "${srcroot}.git" ; then
  git describe --long --abbrev=40 > ${srcroot}VERSION
fi
jemalloc_version=`cat ${srcroot}VERSION`
jemalloc_version_major=`echo ${jemalloc_version} | tr ".g-" " " | awk '{print [$]1}'`
jemalloc_version_minor=`echo ${jemalloc_version} | tr ".g-" " " | awk '{print [$]2}'`
jemalloc_version_bugfix=`echo ${jemalloc_version} | tr ".g-" " " | awk '{print [$]3}'`
jemalloc_version_nrev=`echo ${jemalloc_version} | tr ".g-" " " | awk '{print [$]4}'`
jemalloc_version_gid=`echo ${jemalloc_version} | tr ".g-" " " | awk '{print [$]5}'`
AC_SUBST([jemalloc_version])
AC_SUBST([jemalloc_version_major])
AC_SUBST([jemalloc_version_minor])
AC_SUBST([jemalloc_version_bugfix])
AC_SUBST([jemalloc_version_nrev])
AC_SUBST([jemalloc_version_gid])

dnl ============================================================================
dnl Configure pthreads.

if test "x$abi" != "xpecoff" ; then
  AC_CHECK_HEADERS([pthread.h], , [AC_MSG_ERROR([pthread.h is missing])])
  dnl Some systems may embed pthreads functionality in libc; check for libpthread
  dnl first, but try libc too before failing.
  AC_CHECK_LIB([pthread], [pthread_create], [LIBS="$LIBS -lpthread"],
               [AC_SEARCH_LIBS([pthread_create], , ,
                               AC_MSG_ERROR([libpthread is missing]))])
fi

CPPFLAGS="$CPPFLAGS -D_REENTRANT"

dnl Check whether the BSD-specific _malloc_thread_cleanup() exists.  If so, use
dnl it rather than pthreads TSD cleanup functions to support cleanup during
dnl thread exit, in order to avoid pthreads library recursion during
dnl bootstrapping.
AC_CHECK_FUNC([_malloc_thread_cleanup],
              [have__malloc_thread_cleanup="1"],
              [have__malloc_thread_cleanup="0"]
             )
if test "x$have__malloc_thread_cleanup" = "x1" ; then
  AC_DEFINE([JEMALLOC_MALLOC_THREAD_CLEANUP], [ ])
  force_tls="1"
fi

dnl Check whether the BSD-specific _pthread_mutex_init_calloc_cb() exists.  If
dnl so, mutex initialization causes allocation, and we need to implement this
dnl callback function in order to prevent recursive allocation.
AC_CHECK_FUNC([_pthread_mutex_init_calloc_cb],
              [have__pthread_mutex_init_calloc_cb="1"],
              [have__pthread_mutex_init_calloc_cb="0"]
             )
if test "x$have__pthread_mutex_init_calloc_cb" = "x1" ; then
  AC_DEFINE([JEMALLOC_MUTEX_INIT_CB])
fi

dnl Disable lazy locking by default.
AC_ARG_ENABLE([lazy_lock],
  [AS_HELP_STRING([--enable-lazy-lock],
  [Enable lazy locking (only lock when multi-threaded)])],
[if test "x$enable_lazy_lock" = "xno" ; then
  enable_lazy_lock="0"
else
  enable_lazy_lock="1"
fi
],
[enable_lazy_lock="0"]
)
if test "x$enable_lazy_lock" = "x0" -a "x${force_lazy_lock}" = "x1" ; then
  AC_MSG_RESULT([Forcing lazy-lock to avoid allocator/threading bootstrap issues])
  enable_lazy_lock="1"
fi
if test "x$enable_lazy_lock" = "x1" ; then
  if test "x$abi" != "xpecoff" ; then
    AC_CHECK_HEADERS([dlfcn.h], , [AC_MSG_ERROR([dlfcn.h is missing])])
    AC_CHECK_FUNC([dlsym], [],
      [AC_CHECK_LIB([dl], [dlsym], [LIBS="$LIBS -ldl"],
                    [AC_MSG_ERROR([libdl is missing])])
      ])
  fi
  AC_DEFINE([JEMALLOC_LAZY_LOCK], [ ])
fi
AC_SUBST([enable_lazy_lock])

AC_ARG_ENABLE([tls],
  [AS_HELP_STRING([--disable-tls], [Disable thread-local storage (__thread keyword)])],
if test "x$enable_tls" = "xno" ; then
  enable_tls="0"
else
  enable_tls="1"
fi
,
enable_tls="1"
)
if test "x${enable_tls}" = "x0" -a "x${force_tls}" = "x1" ; then
  AC_MSG_RESULT([Forcing TLS to avoid allocator/threading bootstrap issues])
  enable_tls="1"
fi
if test "x${enable_tls}" = "x1" -a "x${force_tls}" = "x0" ; then
  AC_MSG_RESULT([Forcing no TLS to avoid allocator/threading bootstrap issues])
  enable_tls="0"
fi
if test "x${enable_tls}" = "x1" ; then
AC_MSG_CHECKING([for TLS])
AC_COMPILE_IFELSE([AC_LANG_PROGRAM(
[[
    __thread int x;
]], [[
    x = 42;

    return 0;
]])],
              AC_MSG_RESULT([yes]),
              AC_MSG_RESULT([no])
              enable_tls="0")
fi
AC_SUBST([enable_tls])
if test "x${enable_tls}" = "x1" ; then
  AC_DEFINE_UNQUOTED([JEMALLOC_TLS], [ ])
elif test "x${force_tls}" = "x1" ; then
  AC_MSG_ERROR([Failed to configure TLS, which is mandatory for correct function])
fi

dnl ============================================================================
dnl Check for ffsl(3), and fail if not found.  This function exists on all
dnl platforms that jemalloc currently has a chance of functioning on without
dnl modification.
JE_COMPILABLE([a program using ffsl], [
#include <stdio.h>
#include <strings.h>
#include <string.h>
], [
	{
		int rv = ffsl(0x08);
		printf("%d\n", rv);
	}
], [je_cv_function_ffsl])
if test "x${je_cv_function_ffsl}" != "xyes" ; then
   AC_MSG_ERROR([Cannot build without ffsl(3)])
fi

dnl ============================================================================
dnl Check for atomic(9) operations as provided on FreeBSD.

JE_COMPILABLE([atomic(9)], [
#include <sys/types.h>
#include <machine/atomic.h>
#include <inttypes.h>
], [
	{
		uint32_t x32 = 0;
		volatile uint32_t *x32p = &x32;
		atomic_fetchadd_32(x32p, 1);
	}
	{
		unsigned long xlong = 0;
		volatile unsigned long *xlongp = &xlong;
		atomic_fetchadd_long(xlongp, 1);
	}
], [je_cv_atomic9])
if test "x${je_cv_atomic9}" = "xyes" ; then
  AC_DEFINE([JEMALLOC_ATOMIC9])
fi

dnl ============================================================================
dnl Check for atomic(3) operations as provided on Darwin.

JE_COMPILABLE([Darwin OSAtomic*()], [
#include <libkern/OSAtomic.h>
#include <inttypes.h>
], [
	{
		int32_t x32 = 0;
		volatile int32_t *x32p = &x32;
		OSAtomicAdd32(1, x32p);
	}
	{
		int64_t x64 = 0;
		volatile int64_t *x64p = &x64;
		OSAtomicAdd64(1, x64p);
	}
], [je_cv_osatomic])
if test "x${je_cv_osatomic}" = "xyes" ; then
  AC_DEFINE([JEMALLOC_OSATOMIC], [ ])
fi

dnl ============================================================================
dnl Check whether __sync_{add,sub}_and_fetch() are available despite
dnl __GCC_HAVE_SYNC_COMPARE_AND_SWAP_n macros being undefined.

AC_DEFUN([JE_SYNC_COMPARE_AND_SWAP_CHECK],[
  AC_CACHE_CHECK([whether to force $1-bit __sync_{add,sub}_and_fetch()],
               [je_cv_sync_compare_and_swap_$2],
               [AC_LINK_IFELSE([AC_LANG_PROGRAM([
                                                 #include <stdint.h>
                                                ],
                                                [
                                                 #ifndef __GCC_HAVE_SYNC_COMPARE_AND_SWAP_$2
                                                 {
                                                    uint$1_t x$1 = 0;
                                                    __sync_add_and_fetch(&x$1, 42);
                                                    __sync_sub_and_fetch(&x$1, 1);
                                                 }
                                                 #else
                                                 #error __GCC_HAVE_SYNC_COMPARE_AND_SWAP_$2 is defined, no need to force
                                                 #endif
                                                ])],
                               [je_cv_sync_compare_and_swap_$2=yes],
                               [je_cv_sync_compare_and_swap_$2=no])])

  if test "x${je_cv_sync_compare_and_swap_$2}" = "xyes" ; then
    AC_DEFINE([JE_FORCE_SYNC_COMPARE_AND_SWAP_$2], [ ])
  fi
])

if test "x${je_cv_atomic9}" != "xyes" -a "x${je_cv_osatomic}" != "xyes" ; then
  JE_SYNC_COMPARE_AND_SWAP_CHECK(32, 4)
  JE_SYNC_COMPARE_AND_SWAP_CHECK(64, 8)
fi

dnl ============================================================================
dnl Check for spinlock(3) operations as provided on Darwin.

JE_COMPILABLE([Darwin OSSpin*()], [
#include <libkern/OSAtomic.h>
#include <inttypes.h>
], [
	OSSpinLock lock = 0;
	OSSpinLockLock(&lock);
	OSSpinLockUnlock(&lock);
], [je_cv_osspin])
if test "x${je_cv_osspin}" = "xyes" ; then
  AC_DEFINE([JEMALLOC_OSSPIN], [ ])
fi

dnl ============================================================================
dnl Darwin-related configuration.

AC_ARG_ENABLE([zone-allocator],
  [AS_HELP_STRING([--disable-zone-allocator],
                  [Disable zone allocator for Darwin])],
[if test "x$enable_zone_allocator" = "xno" ; then
  enable_zone_allocator="0"
else
  enable_zone_allocator="1"
fi
],
[if test "x${abi}" = "xmacho"; then
  enable_zone_allocator="1"
fi
]
)
AC_SUBST([enable_zone_allocator])

if test "x${enable_zone_allocator}" = "x1" ; then
  if test "x${abi}" != "xmacho"; then
    AC_MSG_ERROR([--enable-zone-allocator is only supported on Darwin])
  fi
  AC_DEFINE([JEMALLOC_IVSALLOC], [ ])
  AC_DEFINE([JEMALLOC_ZONE], [ ])

  dnl The szone version jumped from 3 to 6 between the OS X 10.5.x and 10.6
  dnl releases.  malloc_zone_t and malloc_introspection_t have new fields in
  dnl 10.6, which is the only source-level indication of the change.
  AC_MSG_CHECKING([malloc zone version])
  AC_DEFUN([JE_ZONE_PROGRAM],
    [AC_LANG_PROGRAM(
      [#include <malloc/malloc.h>],
      [static foo[[sizeof($1) $2 sizeof(void *) * $3 ? 1 : -1]]]
    )])

  AC_COMPILE_IFELSE([JE_ZONE_PROGRAM(malloc_zone_t,==,14)],[JEMALLOC_ZONE_VERSION=3],[
  AC_COMPILE_IFELSE([JE_ZONE_PROGRAM(malloc_zone_t,==,15)],[JEMALLOC_ZONE_VERSION=5],[
  AC_COMPILE_IFELSE([JE_ZONE_PROGRAM(malloc_zone_t,==,16)],[
    AC_COMPILE_IFELSE([JE_ZONE_PROGRAM(malloc_introspection_t,==,9)],[JEMALLOC_ZONE_VERSION=6],[
    AC_COMPILE_IFELSE([JE_ZONE_PROGRAM(malloc_introspection_t,==,13)],[JEMALLOC_ZONE_VERSION=7],[JEMALLOC_ZONE_VERSION=]
  )])],[
  AC_COMPILE_IFELSE([JE_ZONE_PROGRAM(malloc_zone_t,==,17)],[JEMALLOC_ZONE_VERSION=8],[
  AC_COMPILE_IFELSE([JE_ZONE_PROGRAM(malloc_zone_t,>,17)],[JEMALLOC_ZONE_VERSION=9],[JEMALLOC_ZONE_VERSION=]
  )])])])])
  if test "x${JEMALLOC_ZONE_VERSION}" = "x"; then
    AC_MSG_RESULT([unsupported])
    AC_MSG_ERROR([Unsupported malloc zone version])
  fi
  if test "${JEMALLOC_ZONE_VERSION}" = 9; then
    JEMALLOC_ZONE_VERSION=8
    AC_MSG_RESULT([> 8])
  else
    AC_MSG_RESULT([$JEMALLOC_ZONE_VERSION])
  fi
  AC_DEFINE_UNQUOTED(JEMALLOC_ZONE_VERSION, [$JEMALLOC_ZONE_VERSION])
fi

dnl ============================================================================
dnl Check for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL

AC_CONFIG_COMMANDS([include/jemalloc/internal/size_classes.h], [
  mkdir -p "include/jemalloc/internal"
  "${srcdir}/include/jemalloc/internal/size_classes.sh" > "${objroot}include/jemalloc/internal/size_classes.h"
])

dnl Process .in files.
AC_SUBST([cfghdrs_in])
AC_SUBST([cfghdrs_out])
AC_CONFIG_HEADERS([$cfghdrs_tup])

dnl ============================================================================
dnl Generate outputs.
AC_CONFIG_FILES([$cfgoutputs_tup config.stamp bin/jemalloc.sh])
AC_SUBST([cfgoutputs_in])
AC_SUBST([cfgoutputs_out])
AC_OUTPUT

dnl ============================================================================
dnl Print out the results of configuration.
AC_MSG_RESULT([===============================================================================])
AC_MSG_RESULT([jemalloc version   : ${jemalloc_version}])
AC_MSG_RESULT([library revision   : ${rev}])
AC_MSG_RESULT([])
AC_MSG_RESULT([CC                 : ${CC}])
AC_MSG_RESULT([CPPFLAGS           : ${CPPFLAGS}])
AC_MSG_RESULT([CFLAGS             : ${CFLAGS}])
AC_MSG_RESULT([LDFLAGS            : ${LDFLAGS}])
AC_MSG_RESULT([LIBS               : ${LIBS}])
AC_MSG_RESULT([RPATH_EXTRA        : ${RPATH_EXTRA}])
AC_MSG_RESULT([])
AC_MSG_RESULT([XSLTPROC           : ${XSLTPROC}])
AC_MSG_RESULT([XSLROOT            : ${XSLROOT}])
AC_MSG_RESULT([])
AC_MSG_RESULT([PREFIX             : ${PREFIX}])
AC_MSG_RESULT([BINDIR             : ${BINDIR}])
AC_MSG_RESULT([INCLUDEDIR         : ${INCLUDEDIR}])
AC_MSG_RESULT([LIBDIR             : ${LIBDIR}])
AC_MSG_RESULT([DATADIR            : ${DATADIR}])
AC_MSG_RESULT([MANDIR             : ${MANDIR}])
AC_MSG_RESULT([])
AC_MSG_RESULT([srcroot            : ${srcroot}])
AC_MSG_RESULT([abs_srcroot        : ${abs_srcroot}])
AC_MSG_RESULT([objroot            : ${objroot}])
AC_MSG_RESULT([abs_objroot        : ${abs_objroot}])
AC_MSG_RESULT([])
AC_MSG_RESULT([JEMALLOC_PREFIX    : ${JEMALLOC_PREFIX}])
AC_MSG_RESULT([JEMALLOC_PRIVATE_NAMESPACE])
AC_MSG_RESULT([                   : ${JEMALLOC_PRIVATE_NAMESPACE}])
AC_MSG_RESULT([install_suffix     : ${install_suffix}])
AC_MSG_RESULT([autogen            : ${enable_autogen}])
AC_MSG_RESULT([experimental       : ${enable_experimental}])
AC_MSG_RESULT([cc-silence         : ${enable_cc_silence}])
AC_MSG_RESULT([debug              : ${enable_debug}])
AC_MSG_RESULT([stats              : ${enable_stats}])
AC_MSG_RESULT([prof               : ${enable_prof}])
AC_MSG_RESULT([prof-libunwind     : ${enable_prof_libunwind}])
AC_MSG_RESULT([prof-libgcc        : ${enable_prof_libgcc}])
AC_MSG_RESULT([prof-gcc           : ${enable_prof_gcc}])
AC_MSG_RESULT([tcache             : ${enable_tcache}])
AC_MSG_RESULT([fill               : ${enable_fill}])
AC_MSG_RESULT([utrace             : ${enable_utrace}])
AC_MSG_RESULT([valgrind           : ${enable_valgrind}])
AC_MSG_RESULT([xmalloc            : ${enable_xmalloc}])
AC_MSG_RESULT([mremap             : ${enable_mremap}])
AC_MSG_RESULT([munmap             : ${enable_munmap}])
AC_MSG_RESULT([dss                : ${enable_dss}])
AC_MSG_RESULT([lazy_lock          : ${enable_lazy_lock}])
AC_MSG_RESULT([tls                : ${enable_tls}])
AC_MSG_RESULT([===============================================================================])
