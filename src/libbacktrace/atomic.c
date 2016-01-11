/* atomic.c -- Support for atomic functions if not present.
   Copyright (C) 2013-2015 Free Software Foundation, Inc.
   Written by Ian Lance Taylor, Google.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    (1) Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer. 

    (2) Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.  
    
    (3) The name of the author may not be used to
    endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.  */

#include "config.h"

#include <sys/types.h>

#include "backtrace.h"
#include "backtrace-supported.h"
#include "internal.h"

/* This file holds implementations of the atomic functions that are
   used if the host compiler has the sync functions but not the atomic
   functions, as is true of versions of GCC before 4.7.  */

#if !defined (HAVE_ATOMIC_FUNCTIONS) && defined (HAVE_SYNC_FUNCTIONS)

/* Do an atomic load of a pointer.  */

void *
backtrace_atomic_load_pointer (void *arg)
{
  void **pp;
  void *p;

  pp = (void **) arg;
  p = *pp;
  while (!__sync_bool_compare_and_swap (pp, p, p))
    p = *pp;
  return p;
}

/* Do an atomic load of an int.  */

int
backtrace_atomic_load_int (int *p)
{
  int i;

  i = *p;
  while (!__sync_bool_compare_and_swap (p, i, i))
    i = *p;
  return i;
}

/* Do an atomic store of a pointer.  */

void
backtrace_atomic_store_pointer (void *arg, void *p)
{
  void **pp;
  void *old;

  pp = (void **) arg;
  old = *pp;
  while (!__sync_bool_compare_and_swap (pp, old, p))
    old = *pp;
}

/* Do an atomic store of a size_t value.  */

void
backtrace_atomic_store_size_t (size_t *p, size_t v)
{
  size_t old;

  old = *p;
  while (!__sync_bool_compare_and_swap (p, old, v))
    old = *p;
}

/* Do an atomic store of a int value.  */

void
backtrace_atomic_store_int (int *p, int v)
{
  size_t old;

  old = *p;
  while (!__sync_bool_compare_and_swap (p, old, v))
    old = *p;
}

#endif
