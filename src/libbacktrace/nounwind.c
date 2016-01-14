/* backtrace.c -- Entry point for stack backtrace library.
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

#include <sys/types.h>

#include "backtrace.h"

#include "internal.h"

/* This source file is compiled if the unwind library is not
   available.  */

int
backtrace_full (struct backtrace_state *state ATTRIBUTE_UNUSED,
		int skip ATTRIBUTE_UNUSED,
		backtrace_full_callback callback ATTRIBUTE_UNUSED,
		backtrace_error_callback error_callback, void *data)
{
  error_callback (data,
		  "no stack trace because unwind library not available",
		  0);
  return 0;
}

int
backtrace_simple (struct backtrace_state *state ATTRIBUTE_UNUSED,
		  int skip ATTRIBUTE_UNUSED,
		  backtrace_simple_callback callback ATTRIBUTE_UNUSED,
		  backtrace_error_callback error_callback, void *data)
{
  error_callback (data,
		  "no stack trace because unwind library not available",
		  0);
  return 0;
}
