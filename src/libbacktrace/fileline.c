/* fileline.c -- Get file and line number information in a backtrace.
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
#include <sys/stat.h>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>

#define USE_WIN_EXECNAME (defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0600))
#if USE_WIN_EXECNAME
#include <windows.h>
#endif /* USE_WIN_EXECNAME */

#include "backtrace.h"
#include "internal.h"

#ifndef HAVE_GETEXECNAME
#define getexecname() NULL
#endif

/* Targeted fix for Rust backtraces on Windows/GNU.
   We can't set the executable name once during during creation
   of backtrace state for security reasons, see issue #21889 for
   details. So we recalculate it each time the file is accessed.
   We also use QueryFullProcessImageName instead of APIs like
   GetModuleFileName because it's correctly updated when the
   executable is renamed. We alse validate that the opened file
   is indeed our executable by using NtAreMappedFilesTheSame. */
#if USE_WIN_EXECNAME

#undef getexecname
static const char *getexecname(void) {
  /* Accesses to backtrace functionality from Rust are serialized,
     so having a single static unsyncronized buffer is enough. */
  static char buf[MAX_PATH];
  /* The returned name is later passed to `open`, so it needs to
     be encoded in the current locale, so we use `QueryFullProcessImageNameA`.
     As a result paths not representable in the current locale are not supported. */
  DWORD buf_size = MAX_PATH;
  HANDLE process_handle = OpenProcess(PROCESS_QUERY_INFORMATION, FALSE, GetCurrentProcessId());
  return QueryFullProcessImageNameA(process_handle, 0, buf, &buf_size) ? buf : NULL;
}

static int validate_descriptor(int descriptor) {
  FARPROC NtAreMappedFilesTheSame = GetProcAddress(GetModuleHandleW(L"ntdll.dll"),
                                                   "NtAreMappedFilesTheSame");
  if (!NtAreMappedFilesTheSame) {
    return 0;
  }
  HANDLE file_handle = (HANDLE) _get_osfhandle(descriptor);
  if (file_handle == INVALID_HANDLE_VALUE) {
    return 0;
  }
  /* Map the opened file into memory */
  HANDLE file_mapping = CreateFileMappingW(file_handle, NULL, PAGE_READONLY | SEC_IMAGE,
                                           0, 0, NULL);
  if (!file_mapping) {
    return 0;
  }
  LPVOID mapped_view = MapViewOfFile(file_mapping, FILE_MAP_READ, 0, 0, 0);
  if (!mapped_view) {
    CloseHandle(file_mapping);
    return 0;
  }
  /* Now "compare" memory at which the opened file is mapped (mapped_view) with memory
     at which the current executable is mapped (returned by GetModuleHandleW) */
  NTSTATUS status = NtAreMappedFilesTheSame(GetModuleHandleW(NULL), mapped_view);
  UnmapViewOfFile(mapped_view);
  CloseHandle(file_mapping);
  return status == 0;
}

#else /* USE_WIN_EXECNAME */

static int validate_descriptor(int descriptor) {
  return 1;
}

#endif /* USE_WIN_EXECNAME */

/* Initialize the fileline information from the executable.  Returns 1
   on success, 0 on failure.  */

static int
fileline_initialize (struct backtrace_state *state,
		     backtrace_error_callback error_callback, void *data)
{
  int failed;
  fileline fileline_fn;
  int pass;
  int called_error_callback;
  int descriptor;

  if (!state->threaded)
    failed = state->fileline_initialization_failed;
  else
    failed = backtrace_atomic_load_int (&state->fileline_initialization_failed);

  if (failed)
    {
      error_callback (data, "failed to read executable information", -1);
      return 0;
    }

  if (!state->threaded)
    fileline_fn = state->fileline_fn;
  else
    fileline_fn = backtrace_atomic_load_pointer (&state->fileline_fn);
  if (fileline_fn != NULL)
    return 1;

  /* We have not initialized the information.  Do it now.  */

  descriptor = -1;
  called_error_callback = 0;
  for (pass = 0; pass < 4; ++pass)
    {
      const char *filename;
      int does_not_exist;

      switch (pass)
	{
	case 0:
	  filename = state->filename;
	  break;
	case 1:
	  filename = getexecname ();
	  break;
	case 2:
	  filename = "/proc/self/exe";
	  break;
	case 3:
	  filename = "/proc/curproc/file";
	  break;
	default:
	  abort ();
	}

      if (filename == NULL)
	continue;

      descriptor = backtrace_open (filename, error_callback, data,
				   &does_not_exist);
      if (descriptor < 0 && !does_not_exist)
	{
	  called_error_callback = 1;
	  break;
	}
      if (!validate_descriptor(descriptor)) {
        close(descriptor);
        descriptor = -1;
        break;
      }
      if (descriptor >= 0)
	break;
    }

  if (descriptor < 0)
    {
      if (!called_error_callback)
	{
	  if (state->filename != NULL)
	    error_callback (data, state->filename, ENOENT);
	  else
	    error_callback (data,
			    "libbacktrace could not find executable to open",
			    0);
	}
      failed = 1;
    }

  if (!failed)
    {
      if (!backtrace_initialize (state, descriptor, error_callback, data,
				 &fileline_fn))
	failed = 1;
    }

  if (failed)
    {
      if (!state->threaded)
	state->fileline_initialization_failed = 1;
      else
	backtrace_atomic_store_int (&state->fileline_initialization_failed, 1);
      return 0;
    }

  if (!state->threaded)
    state->fileline_fn = fileline_fn;
  else
    {
      backtrace_atomic_store_pointer (&state->fileline_fn, fileline_fn);

      /* Note that if two threads initialize at once, one of the data
	 sets may be leaked.  */
    }

  return 1;
}

/* Given a PC, find the file name, line number, and function name.  */

int
backtrace_pcinfo (struct backtrace_state *state, uintptr_t pc,
		  backtrace_full_callback callback,
		  backtrace_error_callback error_callback, void *data)
{
  if (!fileline_initialize (state, error_callback, data))
    return 0;

  if (state->fileline_initialization_failed)
    return 0;

  return state->fileline_fn (state, pc, callback, error_callback, data);
}

/* Given a PC, find the symbol for it, and its value.  */

int
backtrace_syminfo (struct backtrace_state *state, uintptr_t pc,
		   backtrace_syminfo_callback callback,
		   backtrace_error_callback error_callback, void *data)
{
  if (!fileline_initialize (state, error_callback, data))
    return 0;

  if (state->fileline_initialization_failed)
    return 0;

  state->syminfo_fn (state, pc, callback, error_callback, data);
  return 1;
}
