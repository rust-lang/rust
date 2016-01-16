/* alloc.c -- Memory allocation without mmap.
   Copyright (C) 2012-2016 Free Software Foundation, Inc.
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

#include <errno.h>
#include <stdlib.h>
#include <sys/types.h>

#include "backtrace.h"
#include "internal.h"

/* Allocation routines to use on systems that do not support anonymous
   mmap.  This implementation just uses malloc, which means that the
   backtrace functions may not be safely invoked from a signal
   handler.  */

/* Allocate memory like malloc.  If ERROR_CALLBACK is NULL, don't
   report an error.  */

void *
backtrace_alloc (struct backtrace_state *state ATTRIBUTE_UNUSED,
		 size_t size, backtrace_error_callback error_callback,
		 void *data)
{
  void *ret;

  ret = malloc (size);
  if (ret == NULL)
    {
      if (error_callback)
	error_callback (data, "malloc", errno);
    }
  return ret;
}

/* Free memory.  */

void
backtrace_free (struct backtrace_state *state ATTRIBUTE_UNUSED,
		void *p, size_t size ATTRIBUTE_UNUSED,
		backtrace_error_callback error_callback ATTRIBUTE_UNUSED,
		void *data ATTRIBUTE_UNUSED)
{
  free (p);
}

/* Grow VEC by SIZE bytes.  */

void *
backtrace_vector_grow (struct backtrace_state *state ATTRIBUTE_UNUSED,
		       size_t size, backtrace_error_callback error_callback,
		       void *data, struct backtrace_vector *vec)
{
  void *ret;

  if (size > vec->alc)
    {
      size_t alc;
      void *base;

      if (vec->size == 0)
	alc = 32 * size;
      else if (vec->size >= 4096)
	alc = vec->size + 4096;
      else
	alc = 2 * vec->size;

      if (alc < vec->size + size)
	alc = vec->size + size;

      base = realloc (vec->base, alc);
      if (base == NULL)
	{
	  error_callback (data, "realloc", errno);
	  return NULL;
	}

      vec->base = base;
      vec->alc = alc - vec->size;
    }

  ret = (char *) vec->base + vec->size;
  vec->size += size;
  vec->alc -= size;
  return ret;
}

/* Finish the current allocation on VEC.  */

void *
backtrace_vector_finish (struct backtrace_state *state,
			 struct backtrace_vector *vec,
			 backtrace_error_callback error_callback,
			 void *data)
{
  void *ret;

  /* With this allocator we call realloc in backtrace_vector_grow,
     which means we can't easily reuse the memory here.  So just
     release it.  */
  if (!backtrace_vector_release (state, vec, error_callback, data))
    return NULL;
  ret = vec->base;
  vec->base = NULL;
  vec->size = 0;
  vec->alc = 0;
  return ret;
}

/* Release any extra space allocated for VEC.  */

int
backtrace_vector_release (struct backtrace_state *state ATTRIBUTE_UNUSED,
			  struct backtrace_vector *vec,
			  backtrace_error_callback error_callback,
			  void *data)
{
  vec->base = realloc (vec->base, vec->size);
  if (vec->base == NULL)
    {
      error_callback (data, "realloc", errno);
      return 0;
    }
  vec->alc = 0;
  return 1;
}
