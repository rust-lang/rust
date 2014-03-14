/* Macros for taking apart, interpreting and processing file names.

   These are here because some non-Posix (a.k.a. DOSish) systems have
   drive letter brain-damage at the beginning of an absolute file name,
   use forward- and back-slash in path names interchangeably, and
   some of them have case-insensitive file names.

   Copyright 2000, 2001, 2007, 2010 Free Software Foundation, Inc.

This file is part of BFD, the Binary File Descriptor library.

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

#ifndef FILENAMES_H
#define FILENAMES_H

#include "hashtab.h" /* for hashval_t */

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__MSDOS__) || defined(_WIN32) || defined(__OS2__) || defined (__CYGWIN__)
#  ifndef HAVE_DOS_BASED_FILE_SYSTEM
#    define HAVE_DOS_BASED_FILE_SYSTEM 1
#  endif
#  ifndef HAVE_CASE_INSENSITIVE_FILE_SYSTEM
#    define HAVE_CASE_INSENSITIVE_FILE_SYSTEM 1
#  endif
#  define HAS_DRIVE_SPEC(f) HAS_DOS_DRIVE_SPEC (f)
#  define IS_DIR_SEPARATOR(c) IS_DOS_DIR_SEPARATOR (c)
#  define IS_ABSOLUTE_PATH(f) IS_DOS_ABSOLUTE_PATH (f)
#else /* not DOSish */
#  if defined(__APPLE__)
#    ifndef HAVE_CASE_INSENSITIVE_FILE_SYSTEM
#      define HAVE_CASE_INSENSITIVE_FILE_SYSTEM 1
#    endif
#  endif /* __APPLE__ */
#  define HAS_DRIVE_SPEC(f) (0)
#  define IS_DIR_SEPARATOR(c) IS_UNIX_DIR_SEPARATOR (c)
#  define IS_ABSOLUTE_PATH(f) IS_UNIX_ABSOLUTE_PATH (f)
#endif

#define IS_DIR_SEPARATOR_1(dos_based, c)				\
  (((c) == '/')								\
   || (((c) == '\\') && (dos_based)))

#define HAS_DRIVE_SPEC_1(dos_based, f)			\
  ((f)[0] && ((f)[1] == ':') && (dos_based))

/* Remove the drive spec from F, assuming HAS_DRIVE_SPEC (f).
   The result is a pointer to the remainder of F.  */
#define STRIP_DRIVE_SPEC(f)	((f) + 2)

#define IS_DOS_DIR_SEPARATOR(c) IS_DIR_SEPARATOR_1 (1, c)
#define IS_DOS_ABSOLUTE_PATH(f) IS_ABSOLUTE_PATH_1 (1, f)
#define HAS_DOS_DRIVE_SPEC(f) HAS_DRIVE_SPEC_1 (1, f)

#define IS_UNIX_DIR_SEPARATOR(c) IS_DIR_SEPARATOR_1 (0, c)
#define IS_UNIX_ABSOLUTE_PATH(f) IS_ABSOLUTE_PATH_1 (0, f)

/* Note that when DOS_BASED is true, IS_ABSOLUTE_PATH accepts d:foo as
   well, although it is only semi-absolute.  This is because the users
   of IS_ABSOLUTE_PATH want to know whether to prepend the current
   working directory to a file name, which should not be done with a
   name like d:foo.  */
#define IS_ABSOLUTE_PATH_1(dos_based, f)		 \
  (IS_DIR_SEPARATOR_1 (dos_based, (f)[0])		 \
   || HAS_DRIVE_SPEC_1 (dos_based, f))

extern int filename_cmp (const char *s1, const char *s2);
#define FILENAME_CMP(s1, s2)	filename_cmp(s1, s2)

extern int filename_ncmp (const char *s1, const char *s2,
			  size_t n);

extern hashval_t filename_hash (const void *s);

extern int filename_eq (const void *s1, const void *s2);

#ifdef __cplusplus
}
#endif

#endif /* FILENAMES_H */
