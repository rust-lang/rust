/* ANSI and traditional C compatability macros
   Copyright 1991, 1992, 1993, 1994, 1995, 1996, 1998, 1999, 2000, 2001,
   2002, 2003, 2004, 2005, 2006, 2007, 2009, 2010, 2013
   Free Software Foundation, Inc.
   This file is part of the GNU C Library.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street - Fifth Floor, Boston, MA 02110-1301, USA.  */

/* ANSI and traditional C compatibility macros

   ANSI C is assumed if __STDC__ is #defined.

   Macro		ANSI C definition	Traditional C definition
   -----		---- - ----------	----------- - ----------
   PTR			`void *'		`char *'
   const		not defined		`'
   volatile		not defined		`'
   signed		not defined		`'

   For ease of writing code which uses GCC extensions but needs to be
   portable to other compilers, we provide the GCC_VERSION macro that
   simplifies testing __GNUC__ and __GNUC_MINOR__ together, and various
   wrappers around __attribute__.  Also, __extension__ will be #defined
   to nothing if it doesn't work.  See below.  */

#ifndef	_ANSIDECL_H
#define _ANSIDECL_H	1

#ifdef __cplusplus
extern "C" {
#endif

/* Every source file includes this file,
   so they will all get the switch for lint.  */
/* LINTLIBRARY */

/* Using MACRO(x,y) in cpp #if conditionals does not work with some
   older preprocessors.  Thus we can't define something like this:

#define HAVE_GCC_VERSION(MAJOR, MINOR) \
  (__GNUC__ > (MAJOR) || (__GNUC__ == (MAJOR) && __GNUC_MINOR__ >= (MINOR)))

and then test "#if HAVE_GCC_VERSION(2,7)".

So instead we use the macro below and test it against specific values.  */

/* This macro simplifies testing whether we are using gcc, and if it
   is of a particular minimum version. (Both major & minor numbers are
   significant.)  This macro will evaluate to 0 if we are not using
   gcc at all.  */
#ifndef GCC_VERSION
#define GCC_VERSION (__GNUC__ * 1000 + __GNUC_MINOR__)
#endif /* GCC_VERSION */

#if defined (__STDC__) || defined(__cplusplus) || defined (_AIX) || (defined (__mips) && defined (_SYSTYPE_SVR4)) || defined(_WIN32)
/* All known AIX compilers implement these things (but don't always
   define __STDC__).  The RISC/OS MIPS compiler defines these things
   in SVR4 mode, but does not define __STDC__.  */
/* eraxxon@alumni.rice.edu: The Compaq C++ compiler, unlike many other
   C++ compilers, does not define __STDC__, though it acts as if this
   was so. (Verified versions: 5.7, 6.2, 6.3, 6.5) */

#define PTR		void *

#undef const
#undef volatile
#undef signed

/* inline requires special treatment; it's in C99, and GCC >=2.7 supports
   it too, but it's not in C89.  */
#undef inline
#if __STDC_VERSION__ >= 199901L || defined(__cplusplus) || (defined(__SUNPRO_C) && defined(__C99FEATURES__))
/* it's a keyword */
#else
# if GCC_VERSION >= 2007
#  define inline __inline__   /* __inline__ prevents -pedantic warnings */
# else
#  define inline  /* nothing */
# endif
#endif

#else	/* Not ANSI C.  */

#define PTR		char *

/* some systems define these in header files for non-ansi mode */
#undef const
#undef volatile
#undef signed
#undef inline
#define const
#define volatile
#define signed
#define inline

#endif	/* ANSI C.  */

/* Define macros for some gcc attributes.  This permits us to use the
   macros freely, and know that they will come into play for the
   version of gcc in which they are supported.  */

#if (GCC_VERSION < 2007)
# define __attribute__(x)
#endif

/* Attribute __malloc__ on functions was valid as of gcc 2.96. */
#ifndef ATTRIBUTE_MALLOC
# if (GCC_VERSION >= 2096)
#  define ATTRIBUTE_MALLOC __attribute__ ((__malloc__))
# else
#  define ATTRIBUTE_MALLOC
# endif /* GNUC >= 2.96 */
#endif /* ATTRIBUTE_MALLOC */

/* Attributes on labels were valid as of gcc 2.93 and g++ 4.5.  For
   g++ an attribute on a label must be followed by a semicolon.  */
#ifndef ATTRIBUTE_UNUSED_LABEL
# ifndef __cplusplus
#  if GCC_VERSION >= 2093
#   define ATTRIBUTE_UNUSED_LABEL ATTRIBUTE_UNUSED
#  else
#   define ATTRIBUTE_UNUSED_LABEL
#  endif
# else
#  if GCC_VERSION >= 4005
#   define ATTRIBUTE_UNUSED_LABEL ATTRIBUTE_UNUSED ;
#  else
#   define ATTRIBUTE_UNUSED_LABEL
#  endif
# endif
#endif

/* Similarly to ARG_UNUSED below.  Prior to GCC 3.4, the C++ frontend
   couldn't parse attributes placed after the identifier name, and now
   the entire compiler is built with C++.  */
#ifndef ATTRIBUTE_UNUSED
#if GCC_VERSION >= 3004
#  define ATTRIBUTE_UNUSED __attribute__ ((__unused__))
#else
#define ATTRIBUTE_UNUSED
#endif
#endif /* ATTRIBUTE_UNUSED */

/* Before GCC 3.4, the C++ frontend couldn't parse attributes placed after the
   identifier name.  */
#if ! defined(__cplusplus) || (GCC_VERSION >= 3004)
# define ARG_UNUSED(NAME) NAME ATTRIBUTE_UNUSED
#else /* !__cplusplus || GNUC >= 3.4 */
# define ARG_UNUSED(NAME) NAME
#endif /* !__cplusplus || GNUC >= 3.4 */

#ifndef ATTRIBUTE_NORETURN
#define ATTRIBUTE_NORETURN __attribute__ ((__noreturn__))
#endif /* ATTRIBUTE_NORETURN */

/* Attribute `nonnull' was valid as of gcc 3.3.  */
#ifndef ATTRIBUTE_NONNULL
# if (GCC_VERSION >= 3003)
#  define ATTRIBUTE_NONNULL(m) __attribute__ ((__nonnull__ (m)))
# else
#  define ATTRIBUTE_NONNULL(m)
# endif /* GNUC >= 3.3 */
#endif /* ATTRIBUTE_NONNULL */

/* Attribute `returns_nonnull' was valid as of gcc 4.9.  */
#ifndef ATTRIBUTE_RETURNS_NONNULL
# if (GCC_VERSION >= 4009)
#  define ATTRIBUTE_RETURNS_NONNULL __attribute__ ((__returns_nonnull__))
# else
#  define ATTRIBUTE_RETURNS_NONNULL
# endif /* GNUC >= 4.9 */
#endif /* ATTRIBUTE_RETURNS_NONNULL */

/* Attribute `pure' was valid as of gcc 3.0.  */
#ifndef ATTRIBUTE_PURE
# if (GCC_VERSION >= 3000)
#  define ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define ATTRIBUTE_PURE
# endif /* GNUC >= 3.0 */
#endif /* ATTRIBUTE_PURE */

/* Use ATTRIBUTE_PRINTF when the format specifier must not be NULL.
   This was the case for the `printf' format attribute by itself
   before GCC 3.3, but as of 3.3 we need to add the `nonnull'
   attribute to retain this behavior.  */
#ifndef ATTRIBUTE_PRINTF
#define ATTRIBUTE_PRINTF(m, n) __attribute__ ((__format__ (__printf__, m, n))) ATTRIBUTE_NONNULL(m)
#define ATTRIBUTE_PRINTF_1 ATTRIBUTE_PRINTF(1, 2)
#define ATTRIBUTE_PRINTF_2 ATTRIBUTE_PRINTF(2, 3)
#define ATTRIBUTE_PRINTF_3 ATTRIBUTE_PRINTF(3, 4)
#define ATTRIBUTE_PRINTF_4 ATTRIBUTE_PRINTF(4, 5)
#define ATTRIBUTE_PRINTF_5 ATTRIBUTE_PRINTF(5, 6)
#endif /* ATTRIBUTE_PRINTF */

/* Use ATTRIBUTE_FPTR_PRINTF when the format attribute is to be set on
   a function pointer.  Format attributes were allowed on function
   pointers as of gcc 3.1.  */
#ifndef ATTRIBUTE_FPTR_PRINTF
# if (GCC_VERSION >= 3001)
#  define ATTRIBUTE_FPTR_PRINTF(m, n) ATTRIBUTE_PRINTF(m, n)
# else
#  define ATTRIBUTE_FPTR_PRINTF(m, n)
# endif /* GNUC >= 3.1 */
# define ATTRIBUTE_FPTR_PRINTF_1 ATTRIBUTE_FPTR_PRINTF(1, 2)
# define ATTRIBUTE_FPTR_PRINTF_2 ATTRIBUTE_FPTR_PRINTF(2, 3)
# define ATTRIBUTE_FPTR_PRINTF_3 ATTRIBUTE_FPTR_PRINTF(3, 4)
# define ATTRIBUTE_FPTR_PRINTF_4 ATTRIBUTE_FPTR_PRINTF(4, 5)
# define ATTRIBUTE_FPTR_PRINTF_5 ATTRIBUTE_FPTR_PRINTF(5, 6)
#endif /* ATTRIBUTE_FPTR_PRINTF */

/* Use ATTRIBUTE_NULL_PRINTF when the format specifier may be NULL.  A
   NULL format specifier was allowed as of gcc 3.3.  */
#ifndef ATTRIBUTE_NULL_PRINTF
# if (GCC_VERSION >= 3003)
#  define ATTRIBUTE_NULL_PRINTF(m, n) __attribute__ ((__format__ (__printf__, m, n)))
# else
#  define ATTRIBUTE_NULL_PRINTF(m, n)
# endif /* GNUC >= 3.3 */
# define ATTRIBUTE_NULL_PRINTF_1 ATTRIBUTE_NULL_PRINTF(1, 2)
# define ATTRIBUTE_NULL_PRINTF_2 ATTRIBUTE_NULL_PRINTF(2, 3)
# define ATTRIBUTE_NULL_PRINTF_3 ATTRIBUTE_NULL_PRINTF(3, 4)
# define ATTRIBUTE_NULL_PRINTF_4 ATTRIBUTE_NULL_PRINTF(4, 5)
# define ATTRIBUTE_NULL_PRINTF_5 ATTRIBUTE_NULL_PRINTF(5, 6)
#endif /* ATTRIBUTE_NULL_PRINTF */

/* Attribute `sentinel' was valid as of gcc 3.5.  */
#ifndef ATTRIBUTE_SENTINEL
# if (GCC_VERSION >= 3005)
#  define ATTRIBUTE_SENTINEL __attribute__ ((__sentinel__))
# else
#  define ATTRIBUTE_SENTINEL
# endif /* GNUC >= 3.5 */
#endif /* ATTRIBUTE_SENTINEL */


#ifndef ATTRIBUTE_ALIGNED_ALIGNOF
# if (GCC_VERSION >= 3000)
#  define ATTRIBUTE_ALIGNED_ALIGNOF(m) __attribute__ ((__aligned__ (__alignof__ (m))))
# else
#  define ATTRIBUTE_ALIGNED_ALIGNOF(m)
# endif /* GNUC >= 3.0 */
#endif /* ATTRIBUTE_ALIGNED_ALIGNOF */

/* Useful for structures whose layout must much some binary specification
   regardless of the alignment and padding qualities of the compiler.  */
#ifndef ATTRIBUTE_PACKED
# define ATTRIBUTE_PACKED __attribute__ ((packed))
#endif

/* Attribute `hot' and `cold' was valid as of gcc 4.3.  */
#ifndef ATTRIBUTE_COLD
# if (GCC_VERSION >= 4003)
#  define ATTRIBUTE_COLD __attribute__ ((__cold__))
# else
#  define ATTRIBUTE_COLD
# endif /* GNUC >= 4.3 */
#endif /* ATTRIBUTE_COLD */
#ifndef ATTRIBUTE_HOT
# if (GCC_VERSION >= 4003)
#  define ATTRIBUTE_HOT __attribute__ ((__hot__))
# else
#  define ATTRIBUTE_HOT
# endif /* GNUC >= 4.3 */
#endif /* ATTRIBUTE_HOT */

/* We use __extension__ in some places to suppress -pedantic warnings
   about GCC extensions.  This feature didn't work properly before
   gcc 2.8.  */
#if GCC_VERSION < 2008
#define __extension__
#endif

/* This is used to declare a const variable which should be visible
   outside of the current compilation unit.  Use it as
     EXPORTED_CONST int i = 1;
   This is because the semantics of const are different in C and C++.
   "extern const" is permitted in C but it looks strange, and gcc
   warns about it when -Wc++-compat is not used.  */
#ifdef __cplusplus
#define EXPORTED_CONST extern const
#else
#define EXPORTED_CONST const
#endif

/* Be conservative and only use enum bitfields with C++ or GCC.
   FIXME: provide a complete autoconf test for buggy enum bitfields.  */

#ifdef __cplusplus
#define ENUM_BITFIELD(TYPE) enum TYPE
#elif (GCC_VERSION > 2000)
#define ENUM_BITFIELD(TYPE) __extension__ enum TYPE
#else
#define ENUM_BITFIELD(TYPE) unsigned int
#endif

#ifdef __cplusplus
}
#endif

#endif	/* ansidecl.h	*/
