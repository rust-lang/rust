/* btest.c -- Test for libbacktrace library
   Copyright (C) 2012-2014 Free Software Foundation, Inc.
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

/* This program tests the externally visible interfaces of the
   libbacktrace library.  */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "filenames.h"

#include "backtrace.h"
#include "backtrace-supported.h"

/* Portable attribute syntax.  Actually some of these tests probably
   won't work if the attributes are not recognized.  */

#ifndef GCC_VERSION
# define GCC_VERSION (__GNUC__ * 1000 + __GNUC_MINOR__)
#endif

#if (GCC_VERSION < 2007)
# define __attribute__(x)
#endif

#ifndef ATTRIBUTE_UNUSED
# define ATTRIBUTE_UNUSED __attribute__ ((__unused__))
#endif

/* Used to collect backtrace info.  */

struct info
{
  char *filename;
  int lineno;
  char *function;
};

/* Passed to backtrace callback function.  */

struct bdata
{
  struct info *all;
  size_t index;
  size_t max;
  int failed;
};

/* Passed to backtrace_simple callback function.  */

struct sdata
{
  uintptr_t *addrs;
  size_t index;
  size_t max;
  int failed;
};

/* Passed to backtrace_syminfo callback function.  */

struct symdata
{
  const char *name;
  uintptr_t val, size;
  int failed;
};

/* The backtrace state.  */

static void *state;

/* The number of failures.  */

static int failures;

/* Return the base name in a path.  */

static const char *
base (const char *p)
{
  const char *last;
  const char *s;

  last = NULL;
  for (s = p; *s != '\0'; ++s)
    {
      if (IS_DIR_SEPARATOR (*s))
	last = s + 1;
    }
  return last != NULL ? last : p;
}

/* Check an entry in a struct info array.  */

static void
check (const char *name, int index, const struct info *all, int want_lineno,
       const char *want_function, int *failed)
{
  if (*failed)
    return;
  if (all[index].filename == NULL || all[index].function == NULL)
    {
      fprintf (stderr, "%s: [%d]: missing file name or function name\n",
	       name, index);
      *failed = 1;
      return;
    }
  if (strcmp (base (all[index].filename), "btest.c") != 0)
    {
      fprintf (stderr, "%s: [%d]: got %s expected test.c\n", name, index,
	       all[index].filename);
      *failed = 1;
    }
  if (all[index].lineno != want_lineno)
    {
      fprintf (stderr, "%s: [%d]: got %d expected %d\n", name, index,
	       all[index].lineno, want_lineno);
      *failed = 1;
    }
  if (strcmp (all[index].function, want_function) != 0)
    {
      fprintf (stderr, "%s: [%d]: got %s expected %s\n", name, index,
	       all[index].function, want_function);
      *failed = 1;
    }
}

/* The backtrace callback function.  */

static int
callback_one (void *vdata, uintptr_t pc ATTRIBUTE_UNUSED,
	      const char *filename, int lineno, const char *function)
{
  struct bdata *data = (struct bdata *) vdata;
  struct info *p;

  if (data->index >= data->max)
    {
      fprintf (stderr, "callback_one: callback called too many times\n");
      data->failed = 1;
      return 1;
    }

  p = &data->all[data->index];
  if (filename == NULL)
    p->filename = NULL;
  else
    {
      p->filename = strdup (filename);
      assert (p->filename != NULL);
    }
  p->lineno = lineno;
  if (function == NULL)
    p->function = NULL;
  else
    {
      p->function = strdup (function);
      assert (p->function != NULL);
    }
  ++data->index;

  return 0;
}

/* An error callback passed to backtrace.  */

static void
error_callback_one (void *vdata, const char *msg, int errnum)
{
  struct bdata *data = (struct bdata *) vdata;

  fprintf (stderr, "%s", msg);
  if (errnum > 0)
    fprintf (stderr, ": %s", strerror (errnum));
  fprintf (stderr, "\n");
  data->failed = 1;
}

/* The backtrace_simple callback function.  */

static int
callback_two (void *vdata, uintptr_t pc)
{
  struct sdata *data = (struct sdata *) vdata;

  if (data->index >= data->max)
    {
      fprintf (stderr, "callback_two: callback called too many times\n");
      data->failed = 1;
      return 1;
    }

  data->addrs[data->index] = pc;
  ++data->index;

  return 0;
}

/* An error callback passed to backtrace_simple.  */

static void
error_callback_two (void *vdata, const char *msg, int errnum)
{
  struct sdata *data = (struct sdata *) vdata;

  fprintf (stderr, "%s", msg);
  if (errnum > 0)
    fprintf (stderr, ": %s", strerror (errnum));
  fprintf (stderr, "\n");
  data->failed = 1;
}

/* The backtrace_syminfo callback function.  */

static void
callback_three (void *vdata, uintptr_t pc ATTRIBUTE_UNUSED,
		const char *symname, uintptr_t symval,
		uintptr_t symsize)
{
  struct symdata *data = (struct symdata *) vdata;

  if (symname == NULL)
    data->name = NULL;
  else
    {
      data->name = strdup (symname);
      assert (data->name != NULL);
    }
  data->val = symval;
  data->size = symsize;
}

/* The backtrace_syminfo error callback function.  */

static void
error_callback_three (void *vdata, const char *msg, int errnum)
{
  struct symdata *data = (struct symdata *) vdata;

  fprintf (stderr, "%s", msg);
  if (errnum > 0)
    fprintf (stderr, ": %s", strerror (errnum));
  fprintf (stderr, "\n");
  data->failed = 1;
}

/* Test the backtrace function with non-inlined functions.  */

static int test1 (void) __attribute__ ((noinline, unused));
static int f2 (int) __attribute__ ((noinline));
static int f3 (int, int) __attribute__ ((noinline));

static int
test1 (void)
{
  /* Returning a value here and elsewhere avoids a tailcall which
     would mess up the backtrace.  */
  return f2 (__LINE__) + 1;
}

static int
f2 (int f1line)
{
  return f3 (f1line, __LINE__) + 2;
}

static int
f3 (int f1line, int f2line)
{
  struct info all[20];
  struct bdata data;
  int f3line;
  int i;

  data.all = &all[0];
  data.index = 0;
  data.max = 20;
  data.failed = 0;

  f3line = __LINE__ + 1;
  i = backtrace_full (state, 0, callback_one, error_callback_one, &data);

  if (i != 0)
    {
      fprintf (stderr, "test1: unexpected return value %d\n", i);
      data.failed = 1;
    }

  if (data.index < 3)
    {
      fprintf (stderr,
	       "test1: not enough frames; got %zu, expected at least 3\n",
	       data.index);
      data.failed = 1;
    }

  check ("test1", 0, all, f3line, "f3", &data.failed);
  check ("test1", 1, all, f2line, "f2", &data.failed);
  check ("test1", 2, all, f1line, "test1", &data.failed);

  printf ("%s: backtrace_full noinline\n", data.failed ? "FAIL" : "PASS");

  if (data.failed)
    ++failures;

  return failures;
}

/* Test the backtrace function with inlined functions.  */

static inline int test2 (void) __attribute__ ((always_inline, unused));
static inline int f12 (int) __attribute__ ((always_inline));
static inline int f13 (int, int) __attribute__ ((always_inline));

static inline int
test2 (void)
{
  return f12 (__LINE__) + 1;
}

static inline int
f12 (int f1line)
{
  return f13 (f1line, __LINE__) + 2;
}

static inline int
f13 (int f1line, int f2line)
{
  struct info all[20];
  struct bdata data;
  int f3line;
  int i;

  data.all = &all[0];
  data.index = 0;
  data.max = 20;
  data.failed = 0;

  f3line = __LINE__ + 1;
  i = backtrace_full (state, 0, callback_one, error_callback_one, &data);

  if (i != 0)
    {
      fprintf (stderr, "test2: unexpected return value %d\n", i);
      data.failed = 1;
    }

  check ("test2", 0, all, f3line, "f13", &data.failed);
  check ("test2", 1, all, f2line, "f12", &data.failed);
  check ("test2", 2, all, f1line, "test2", &data.failed);

  printf ("%s: backtrace_full inline\n", data.failed ? "FAIL" : "PASS");

  if (data.failed)
    ++failures;

  return failures;
}

/* Test the backtrace_simple function with non-inlined functions.  */

static int test3 (void) __attribute__ ((noinline, unused));
static int f22 (int) __attribute__ ((noinline));
static int f23 (int, int) __attribute__ ((noinline));

static int
test3 (void)
{
  return f22 (__LINE__) + 1;
}

static int
f22 (int f1line)
{
  return f23 (f1line, __LINE__) + 2;
}

static int
f23 (int f1line, int f2line)
{
  uintptr_t addrs[20];
  struct sdata data;
  int f3line;
  int i;

  data.addrs = &addrs[0];
  data.index = 0;
  data.max = 20;
  data.failed = 0;

  f3line = __LINE__ + 1;
  i = backtrace_simple (state, 0, callback_two, error_callback_two, &data);

  if (i != 0)
    {
      fprintf (stderr, "test3: unexpected return value %d\n", i);
      data.failed = 1;
    }

  if (!data.failed)
    {
      struct info all[20];
      struct bdata bdata;
      int j;

      bdata.all = &all[0];
      bdata.index = 0;
      bdata.max = 20;
      bdata.failed = 0;

      for (j = 0; j < 3; ++j)
	{
	  i = backtrace_pcinfo (state, addrs[j], callback_one,
				error_callback_one, &bdata);
	  if (i != 0)
	    {
	      fprintf (stderr,
		       ("test3: unexpected return value "
			"from backtrace_pcinfo %d\n"),
		       i);
	      bdata.failed = 1;
	    }
	  if (!bdata.failed && bdata.index != (size_t) (j + 1))
	    {
	      fprintf (stderr,
		       ("wrong number of calls from backtrace_pcinfo "
			"got %u expected %d\n"),
		       (unsigned int) bdata.index, j + 1);
	      bdata.failed = 1;
	    }
	}      

      check ("test3", 0, all, f3line, "f23", &bdata.failed);
      check ("test3", 1, all, f2line, "f22", &bdata.failed);
      check ("test3", 2, all, f1line, "test3", &bdata.failed);

      if (bdata.failed)
	data.failed = 1;

      for (j = 0; j < 3; ++j)
	{
	  struct symdata symdata;

	  symdata.name = NULL;
	  symdata.val = 0;
	  symdata.size = 0;
	  symdata.failed = 0;

	  i = backtrace_syminfo (state, addrs[j], callback_three,
				 error_callback_three, &symdata);
	  if (i == 0)
	    {
	      fprintf (stderr,
		       ("test3: [%d]: unexpected return value "
			"from backtrace_syminfo %d\n"),
		       j, i);
	      symdata.failed = 1;
	    }

	  if (!symdata.failed)
	    {
	      const char *expected;

	      switch (j)
		{
		case 0:
		  expected = "f23";
		  break;
		case 1:
		  expected = "f22";
		  break;
		case 2:
		  expected = "test3";
		  break;
		default:
		  assert (0);
		}

	      if (symdata.name == NULL)
		{
		  fprintf (stderr, "test3: [%d]: NULL syminfo name\n", j);
		  symdata.failed = 1;
		}
	      /* Use strncmp, not strcmp, because GCC might create a
		 clone.  */
	      else if (strncmp (symdata.name, expected, strlen (expected))
		       != 0)
		{
		  fprintf (stderr,
			   ("test3: [%d]: unexpected syminfo name "
			    "got %s expected %s\n"),
			   j, symdata.name, expected);
		  symdata.failed = 1;
		}
	    }

	  if (symdata.failed)
	    data.failed = 1;
	}
    }

  printf ("%s: backtrace_simple noinline\n", data.failed ? "FAIL" : "PASS");

  if (data.failed)
    ++failures;

  return failures;
}

/* Test the backtrace_simple function with inlined functions.  */

static inline int test4 (void) __attribute__ ((always_inline, unused));
static inline int f32 (int) __attribute__ ((always_inline));
static inline int f33 (int, int) __attribute__ ((always_inline));

static inline int
test4 (void)
{
  return f32 (__LINE__) + 1;
}

static inline int
f32 (int f1line)
{
  return f33 (f1line, __LINE__) + 2;
}

static inline int
f33 (int f1line, int f2line)
{
  uintptr_t addrs[20];
  struct sdata data;
  int f3line;
  int i;

  data.addrs = &addrs[0];
  data.index = 0;
  data.max = 20;
  data.failed = 0;

  f3line = __LINE__ + 1;
  i = backtrace_simple (state, 0, callback_two, error_callback_two, &data);

  if (i != 0)
    {
      fprintf (stderr, "test3: unexpected return value %d\n", i);
      data.failed = 1;
    }

  if (!data.failed)
    {
      struct info all[20];
      struct bdata bdata;

      bdata.all = &all[0];
      bdata.index = 0;
      bdata.max = 20;
      bdata.failed = 0;

      i = backtrace_pcinfo (state, addrs[0], callback_one, error_callback_one,
			    &bdata);
      if (i != 0)
	{
	  fprintf (stderr,
		   ("test4: unexpected return value "
		    "from backtrace_pcinfo %d\n"),
		   i);
	  bdata.failed = 1;
	}

      check ("test4", 0, all, f3line, "f33", &bdata.failed);
      check ("test4", 1, all, f2line, "f32", &bdata.failed);
      check ("test4", 2, all, f1line, "test4", &bdata.failed);

      if (bdata.failed)
	data.failed = 1;
    }

  printf ("%s: backtrace_simple inline\n", data.failed ? "FAIL" : "PASS");

  if (data.failed)
    ++failures;

  return failures;
}

int global = 1;

static int
test5 (void)
{
  struct symdata symdata;
  int i;
  uintptr_t addr = (uintptr_t) &global;

  if (sizeof (global) > 1)
    addr += 1;

  symdata.name = NULL;
  symdata.val = 0;
  symdata.size = 0;
  symdata.failed = 0;

  i = backtrace_syminfo (state, addr, callback_three,
			 error_callback_three, &symdata);
  if (i == 0)
    {
      fprintf (stderr,
	       "test5: unexpected return value from backtrace_syminfo %d\n",
	       i);
      symdata.failed = 1;
    }

  if (!symdata.failed)
    {
      if (symdata.name == NULL)
	{
	  fprintf (stderr, "test5: NULL syminfo name\n");
	  symdata.failed = 1;
	}
      else if (strcmp (symdata.name, "global") != 0)
	{
	  fprintf (stderr,
		   "test5: unexpected syminfo name got %s expected %s\n",
		   symdata.name, "global");
	  symdata.failed = 1;
	}
      else if (symdata.val != (uintptr_t) &global)
	{
	  fprintf (stderr,
		   "test5: unexpected syminfo value got %lx expected %lx\n",
		   (unsigned long) symdata.val,
		   (unsigned long) (uintptr_t) &global);
	  symdata.failed = 1;
	}
      else if (symdata.size != sizeof (global))
	{
	  fprintf (stderr,
		   "test5: unexpected syminfo size got %lx expected %lx\n",
		   (unsigned long) symdata.size,
		   (unsigned long) sizeof (global));
	  symdata.failed = 1;
	}
    }

  printf ("%s: backtrace_syminfo variable\n",
	  symdata.failed ? "FAIL" : "PASS");

  if (symdata.failed)
    ++failures;

  return failures;
}

static void
error_callback_create (void *data ATTRIBUTE_UNUSED, const char *msg,
		       int errnum)
{
  fprintf (stderr, "%s", msg);
  if (errnum > 0)
    fprintf (stderr, ": %s", strerror (errnum));
  fprintf (stderr, "\n");
  exit (EXIT_FAILURE);
}

/* Run all the tests.  */

int
main (int argc ATTRIBUTE_UNUSED, char **argv)
{
  state = backtrace_create_state (argv[0], BACKTRACE_SUPPORTS_THREADS,
				  error_callback_create, NULL);

#if BACKTRACE_SUPPORTED
  test1 ();
  test2 ();
  test3 ();
  test4 ();
  test5 ();
#endif

  exit (failures ? EXIT_FAILURE : EXIT_SUCCESS);
}
