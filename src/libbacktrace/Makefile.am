# Makefile.am -- Backtrace Makefile.
# Copyright (C) 2012-2015 Free Software Foundation, Inc.

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

ACLOCAL_AMFLAGS = -I .. -I ../config

AM_CPPFLAGS = -I $(top_srcdir)/../include -I $(top_srcdir)/../libgcc \
	-I ../libgcc

AM_CFLAGS = $(EXTRA_FLAGS) $(WARN_FLAGS) $(PIC_FLAG)

noinst_LTLIBRARIES = libbacktrace.la

libbacktrace_la_SOURCES = \
	backtrace.h \
	atomic.c \
	dwarf.c \
	fileline.c \
	internal.h \
	posix.c \
	print.c \
	sort.c \
	state.c

BACKTRACE_FILES = \
	backtrace.c \
	simple.c \
	nounwind.c

FORMAT_FILES = \
	elf.c \
	pecoff.c \
	unknown.c

VIEW_FILES = \
	read.c \
	mmapio.c

ALLOC_FILES = \
	alloc.c \
	mmap.c

EXTRA_libbacktrace_la_SOURCES = \
	$(BACKTRACE_FILES) \
	$(FORMAT_FILES) \
	$(VIEW_FILES) \
	$(ALLOC_FILES)

libbacktrace_la_LIBADD = \
	$(BACKTRACE_FILE) \
	$(FORMAT_FILE) \
	$(VIEW_FILE) \
	$(ALLOC_FILE)

libbacktrace_la_DEPENDENCIES = $(libbacktrace_la_LIBADD)

# Testsuite.

check_PROGRAMS =

TESTS = $(check_PROGRAMS)

if NATIVE

btest_SOURCES = btest.c
btest_CFLAGS = $(AM_CFLAGS) -g -O
btest_LDADD = libbacktrace.la

check_PROGRAMS += btest

stest_SOURCES = stest.c
stest_LDADD = libbacktrace.la

check_PROGRAMS += stest

endif NATIVE

# We can't use automake's automatic dependency tracking, because it
# breaks when using bootstrap-lean.  Automatic dependency tracking
# with GCC bootstrap will cause some of the objects to depend on
# header files in prev-gcc/include, e.g., stddef.h and stdarg.h.  When
# using bootstrap-lean, prev-gcc is removed after each stage.  When
# running "make install", those header files will be gone, causing the
# library to be rebuilt at install time.  That may not succeed.

# These manual dependencies do not include dependencies on unwind.h,
# even though that is part of GCC, because where to find it depends on
# whether we are being built as a host library or a target library.

INCDIR = $(top_srcdir)/../include
alloc.lo: config.h backtrace.h internal.h
backtrace.lo: config.h backtrace.h
btest.lo: (INCDIR)/filenames.h backtrace.h backtrace-supported.h
dwarf.lo: config.h $(INCDIR)/dwarf2.h $(INCDIR)/dwarf2.def \
	$(INCDIR)/filenames.h backtrace.h internal.h
elf.lo: config.h backtrace.h internal.h
fileline.lo: config.h backtrace.h internal.h
mmap.lo: config.h backtrace.h internal.h
mmapio.lo: config.h backtrace.h internal.h
nounwind.lo: config.h internal.h
pecoff.lo: config.h backtrace.h internal.h
posix.lo: config.h backtrace.h internal.h
print.lo: config.h backtrace.h internal.h
read.lo: config.h backtrace.h internal.h
simple.lo: config.h backtrace.h internal.h
state.lo: config.h backtrace.h backtrace-supported.h internal.h
unknown.lo: config.h backtrace.h internal.h
