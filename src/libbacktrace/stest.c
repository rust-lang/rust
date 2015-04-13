/* stest.c -- Test for libbacktrace internal sort function
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include "backtrace.h"
#include "internal.h"

/* Test the local qsort implementation.  */

#define MAX 10

struct test
{
  size_t count;
  int input[MAX];
  int output[MAX];
};

static struct test tests[] =
  {
    {
      10,
      { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
      { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }
    },
    {
      9,
      { 1, 2, 3, 4, 5, 6, 7, 8, 9 },
      { 1, 2, 3, 4, 5, 6, 7, 8, 9 }
    },
    {
      10,
      { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 },
      { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
    },
    {
      9,
      { 9, 8, 7, 6, 5, 4, 3, 2, 1 },
      { 1, 2, 3, 4, 5, 6, 7, 8, 9 },
    },
    {
      10,
      { 2, 4, 6, 8, 10, 1, 3, 5, 7, 9 },
      { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
    },
    {
      5,
      { 4, 5, 3, 1, 2 },
      { 1, 2, 3, 4, 5 },
    },
    {
      5,
      { 1, 1, 1, 1, 1 },
      { 1, 1, 1, 1, 1 },
    },
    {
      5,
      { 1, 1, 2, 1, 1 },
      { 1, 1, 1, 1, 2 },
    },
    {
      5,
      { 2, 1, 1, 1, 1 },
      { 1, 1, 1, 1, 2 },
    },
  };

static int
compare (const void *a, const void *b)
{
  const int *ai = (const int *) a;
  const int *bi = (const int *) b;

  return *ai - *bi;
}

int
main (int argc ATTRIBUTE_UNUSED, char **argv ATTRIBUTE_UNUSED)
{
  int failures;
  size_t i;
  int a[MAX];

  failures = 0;
  for (i = 0; i < sizeof tests / sizeof tests[0]; i++)
    {
      memcpy (a, tests[i].input, tests[i].count * sizeof (int));
      backtrace_qsort (a, tests[i].count, sizeof (int), compare);
      if (memcmp (a, tests[i].output, tests[i].count * sizeof (int)) != 0)
	{
	  size_t j;

	  fprintf (stderr, "test %d failed:", (int) i);
	  for (j = 0; j < tests[i].count; j++)
	    fprintf (stderr, " %d", a[j]);
	  fprintf (stderr, "\n");
	  ++failures;
	}
    }

  exit (failures > 0 ? EXIT_FAILURE : EXIT_SUCCESS);
}
