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

#include "unwind.h"
#include "backtrace.h"
#include "internal.h"

/* The main backtrace_full routine.  */

/* Data passed through _Unwind_Backtrace.  */

struct backtrace_data
{
  /* Number of frames to skip.  */
  int skip;
  /* Library state.  */
  struct backtrace_state *state;
  /* Callback routine.  */
  backtrace_full_callback callback;
  /* Error callback routine.  */
  backtrace_error_callback error_callback;
  /* Data to pass to callback routines.  */
  void *data;
  /* Value to return from backtrace_full.  */
  int ret;
  /* Whether there is any memory available.  */
  int can_alloc;
};

/* Unwind library callback routine.  This is passed to
   _Unwind_Backtrace.  */

static _Unwind_Reason_Code
unwind (struct _Unwind_Context *context, void *vdata)
{
  struct backtrace_data *bdata = (struct backtrace_data *) vdata;
  uintptr_t pc;
  int ip_before_insn = 0;

#ifdef HAVE_GETIPINFO
  pc = _Unwind_GetIPInfo (context, &ip_before_insn);
#else
  pc = _Unwind_GetIP (context);
#endif

  if (bdata->skip > 0)
    {
      --bdata->skip;
      return _URC_NO_REASON;
    }

  if (!ip_before_insn)
    --pc;

  if (!bdata->can_alloc)
    bdata->ret = bdata->callback (bdata->data, pc, NULL, 0, NULL);
  else
    bdata->ret = backtrace_pcinfo (bdata->state, pc, bdata->callback,
				   bdata->error_callback, bdata->data);
  if (bdata->ret != 0)
    return _URC_END_OF_STACK;

  return _URC_NO_REASON;
}

/* Get a stack backtrace.  */

int
backtrace_full (struct backtrace_state *state, int skip,
		backtrace_full_callback callback,
		backtrace_error_callback error_callback, void *data)
{
  struct backtrace_data bdata;
  void *p;

  bdata.skip = skip + 1;
  bdata.state = state;
  bdata.callback = callback;
  bdata.error_callback = error_callback;
  bdata.data = data;
  bdata.ret = 0;

  /* If we can't allocate any memory at all, don't try to produce
     file/line information.  */
  p = backtrace_alloc (state, 4096, NULL, NULL);
  if (p == NULL)
    bdata.can_alloc = 0;
  else
    {
      backtrace_free (state, p, 4096, NULL, NULL);
      bdata.can_alloc = 1;
    }

  _Unwind_Backtrace (unwind, &bdata);
  return bdata.ret;
}
