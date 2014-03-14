/* backtrace.h -- Public header file for stack backtrace library.
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

#ifndef BACKTRACE_H
#define BACKTRACE_H

#include <stddef.h>
#include <stdio.h>

/* We want to get a definition for uintptr_t, but we still care about
   systems that don't have <stdint.h>.  */
#if defined(__GLIBC__) && __GLIBC__ >= 2

#include <stdint.h>

#elif defined(HAVE_STDINT_H)

#include <stdint.h>

#else

/* Systems that don't have <stdint.h> must provide gstdint.h, e.g.,
   from GCC_HEADER_STDINT in configure.ac.  */
#include "gstdint.h"

#endif

#ifdef __cplusplus
extern "C" {
#endif

/* The backtrace state.  This struct is intentionally not defined in
   the public interface.  */

struct backtrace_state;

/* The type of the error callback argument to backtrace functions.
   This function, if not NULL, will be called for certain error cases.
   The DATA argument is passed to the function that calls this one.
   The MSG argument is an error message.  The ERRNUM argument, if
   greater than 0, holds an errno value.  The MSG buffer may become
   invalid after this function returns.

   As a special case, the ERRNUM argument will be passed as -1 if no
   debug info can be found for the executable, but the function
   requires debug info (e.g., backtrace_full, backtrace_pcinfo).  The
   MSG in this case will be something along the lines of "no debug
   info".  Similarly, ERRNUM will be passed as -1 if there is no
   symbol table, but the function requires a symbol table (e.g.,
   backtrace_syminfo).  This may be used as a signal that some other
   approach should be tried.  */

typedef void (*backtrace_error_callback) (void *data, const char *msg,
					  int errnum);

/* Create state information for the backtrace routines.  This must be
   called before any of the other routines, and its return value must
   be passed to all of the other routines.  FILENAME is the path name
   of the executable file; if it is NULL the library will try
   system-specific path names.  If not NULL, FILENAME must point to a
   permanent buffer.  If THREADED is non-zero the state may be
   accessed by multiple threads simultaneously, and the library will
   use appropriate atomic operations.  If THREADED is zero the state
   may only be accessed by one thread at a time.  This returns a state
   pointer on success, NULL on error.  If an error occurs, this will
   call the ERROR_CALLBACK routine.  */

extern struct backtrace_state *backtrace_create_state (
    const char *filename, int threaded,
    backtrace_error_callback error_callback, void *data);

/* The type of the callback argument to the backtrace_full function.
   DATA is the argument passed to backtrace_full.  PC is the program
   counter.  FILENAME is the name of the file containing PC, or NULL
   if not available.  LINENO is the line number in FILENAME containing
   PC, or 0 if not available.  FUNCTION is the name of the function
   containing PC, or NULL if not available.  This should return 0 to
   continuing tracing.  The FILENAME and FUNCTION buffers may become
   invalid after this function returns.  */

typedef int (*backtrace_full_callback) (void *data, uintptr_t pc,
					const char *filename, int lineno,
					const char *function);

/* Get a full stack backtrace.  SKIP is the number of frames to skip;
   passing 0 will start the trace with the function calling
   backtrace_full.  DATA is passed to the callback routine.  If any
   call to CALLBACK returns a non-zero value, the stack backtrace
   stops, and backtrace returns that value; this may be used to limit
   the number of stack frames desired.  If all calls to CALLBACK
   return 0, backtrace returns 0.  The backtrace_full function will
   make at least one call to either CALLBACK or ERROR_CALLBACK.  This
   function requires debug info for the executable.  */

extern int backtrace_full (struct backtrace_state *state, int skip,
			   backtrace_full_callback callback,
			   backtrace_error_callback error_callback,
			   void *data);

/* The type of the callback argument to the backtrace_simple function.
   DATA is the argument passed to simple_backtrace.  PC is the program
   counter.  This should return 0 to continue tracing.  */

typedef int (*backtrace_simple_callback) (void *data, uintptr_t pc);

/* Get a simple backtrace.  SKIP is the number of frames to skip, as
   in backtrace.  DATA is passed to the callback routine.  If any call
   to CALLBACK returns a non-zero value, the stack backtrace stops,
   and backtrace_simple returns that value.  Otherwise
   backtrace_simple returns 0.  The backtrace_simple function will
   make at least one call to either CALLBACK or ERROR_CALLBACK.  This
   function does not require any debug info for the executable.  */

extern int backtrace_simple (struct backtrace_state *state, int skip,
			     backtrace_simple_callback callback,
			     backtrace_error_callback error_callback,
			     void *data);

/* Print the current backtrace in a user readable format to a FILE.
   SKIP is the number of frames to skip, as in backtrace_full.  Any
   error messages are printed to stderr.  This function requires debug
   info for the executable.  */

extern void backtrace_print (struct backtrace_state *state, int skip, FILE *);

/* Given PC, a program counter in the current program, call the
   callback function with filename, line number, and function name
   information.  This will normally call the callback function exactly
   once.  However, if the PC happens to describe an inlined call, and
   the debugging information contains the necessary information, then
   this may call the callback function multiple times.  This will make
   at least one call to either CALLBACK or ERROR_CALLBACK.  This
   returns the first non-zero value returned by CALLBACK, or 0.  */

extern int backtrace_pcinfo (struct backtrace_state *state, uintptr_t pc,
			     backtrace_full_callback callback,
			     backtrace_error_callback error_callback,
			     void *data);

/* The type of the callback argument to backtrace_syminfo.  DATA and
   PC are the arguments passed to backtrace_syminfo.  SYMNAME is the
   name of the symbol for the corresponding code.  SYMVAL is the
   value and SYMSIZE is the size of the symbol.  SYMNAME will be NULL
   if no error occurred but the symbol could not be found.  */

typedef void (*backtrace_syminfo_callback) (void *data, uintptr_t pc,
					    const char *symname,
					    uintptr_t symval,
					    uintptr_t symsize);

/* Given ADDR, an address or program counter in the current program,
   call the callback information with the symbol name and value
   describing the function or variable in which ADDR may be found.
   This will call either CALLBACK or ERROR_CALLBACK exactly once.
   This returns 1 on success, 0 on failure.  This function requires
   the symbol table but does not require the debug info.  Note that if
   the symbol table is present but ADDR could not be found in the
   table, CALLBACK will be called with a NULL SYMNAME argument.
   Returns 1 on success, 0 on error.  */

extern int backtrace_syminfo (struct backtrace_state *state, uintptr_t addr,
			      backtrace_syminfo_callback callback,
			      backtrace_error_callback error_callback,
			      void *data);

#ifdef __cplusplus
} /* End extern "C".  */
#endif

#endif
