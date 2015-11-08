/* sort.c -- Sort without allocating memory
   Copyright (C) 2012-2015 Free Software Foundation, Inc.
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

#include <stddef.h>
#include <sys/types.h>

#include "backtrace.h"
#include "internal.h"

/* The GNU glibc version of qsort allocates memory, which we must not
   do if we are invoked by a signal handler.  So provide our own
   sort.  */

static void
swap (char *a, char *b, size_t size)
{
  size_t i;

  for (i = 0; i < size; i++, a++, b++)
    {
      char t;

      t = *a;
      *a = *b;
      *b = t;
    }
}

void
backtrace_qsort (void *basearg, size_t count, size_t size,
		 int (*compar) (const void *, const void *))
{
  char *base = (char *) basearg;
  size_t i;
  size_t mid;

 tail_recurse:
  if (count < 2)
    return;

  /* The symbol table and DWARF tables, which is all we use this
     routine for, tend to be roughly sorted.  Pick the middle element
     in the array as our pivot point, so that we are more likely to
     cut the array in half for each recursion step.  */
  swap (base, base + (count / 2) * size, size);

  mid = 0;
  for (i = 1; i < count; i++)
    {
      if ((*compar) (base, base + i * size) > 0)
	{
	  ++mid;
	  if (i != mid)
	    swap (base + mid * size, base + i * size, size);
	}
    }

  if (mid > 0)
    swap (base, base + mid * size, size);

  /* Recurse with the smaller array, loop with the larger one.  That
     ensures that our maximum stack depth is log count.  */
  if (2 * mid < count)
    {
      backtrace_qsort (base, mid, size, compar);
      base += (mid + 1) * size;
      count -= mid + 1;
      goto tail_recurse;
    }
  else
    {
      backtrace_qsort (base + (mid + 1) * size, count - (mid + 1),
		       size, compar);
      count = mid;
      goto tail_recurse;
    }
}
