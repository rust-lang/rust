/* print.c -- Print the current backtrace.
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

#include "config.h"

#include <stdio.h>
#include <string.h>
#include <sys/types.h>

#include "backtrace.h"
#include "internal.h"

/* Passed to callbacks.  */

struct print_data
{
  struct backtrace_state *state;
  FILE *f;
};

/* Print one level of a backtrace.  */

static int
print_callback (void *data, uintptr_t pc, const char *filename, int lineno,
		const char *function)
{
  struct print_data *pdata = (struct print_data *) data;

  fprintf (pdata->f, "0x%lx %s\n\t%s:%d\n",
	   (unsigned long) pc,
	   function == NULL ? "???" : function,
	   filename == NULL ? "???" : filename,
	   lineno);
  return 0;
}

/* Print errors to stderr.  */

static void
error_callback (void *data, const char *msg, int errnum)
{
  struct print_data *pdata = (struct print_data *) data;

  if (pdata->state->filename != NULL)
    fprintf (stderr, "%s: ", pdata->state->filename);
  fprintf (stderr, "libbacktrace: %s", msg);
  if (errnum > 0)
    fprintf (stderr, ": %s", strerror (errnum));
  fputc ('\n', stderr);
}

/* Print a backtrace.  */

void
backtrace_print (struct backtrace_state *state, int skip, FILE *f)
{
  struct print_data data;

  data.state = state;
  data.f = f;
  backtrace_full (state, skip + 1, print_callback, error_callback,
		  (void *) &data);
}
