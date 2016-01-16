/* internal.h -- Internal header file for stack backtrace library.
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

#ifndef BACKTRACE_INTERNAL_H
#define BACKTRACE_INTERNAL_H

/* We assume that <sys/types.h> and "backtrace.h" have already been
   included.  */

#ifndef GCC_VERSION
# define GCC_VERSION (__GNUC__ * 1000 + __GNUC_MINOR__)
#endif

#if (GCC_VERSION < 2007)
# define __attribute__(x)
#endif

#ifndef ATTRIBUTE_UNUSED
# define ATTRIBUTE_UNUSED __attribute__ ((__unused__))
#endif

#ifndef ATTRIBUTE_MALLOC
# if (GCC_VERSION >= 2096)
#  define ATTRIBUTE_MALLOC __attribute__ ((__malloc__))
# else
#  define ATTRIBUTE_MALLOC
# endif
#endif

#ifndef HAVE_SYNC_FUNCTIONS

/* Define out the sync functions.  These should never be called if
   they are not available.  */

#define __sync_bool_compare_and_swap(A, B, C) (abort(), 1)
#define __sync_lock_test_and_set(A, B) (abort(), 0)
#define __sync_lock_release(A) abort()

#endif /* !defined (HAVE_SYNC_FUNCTIONS) */

#ifdef HAVE_ATOMIC_FUNCTIONS

/* We have the atomic builtin functions.  */

#define backtrace_atomic_load_pointer(p) \
    __atomic_load_n ((p), __ATOMIC_ACQUIRE)
#define backtrace_atomic_load_int(p) \
    __atomic_load_n ((p), __ATOMIC_ACQUIRE)
#define backtrace_atomic_store_pointer(p, v) \
    __atomic_store_n ((p), (v), __ATOMIC_RELEASE)
#define backtrace_atomic_store_size_t(p, v) \
    __atomic_store_n ((p), (v), __ATOMIC_RELEASE)
#define backtrace_atomic_store_int(p, v) \
    __atomic_store_n ((p), (v), __ATOMIC_RELEASE)

#else /* !defined (HAVE_ATOMIC_FUNCTIONS) */
#ifdef HAVE_SYNC_FUNCTIONS

/* We have the sync functions but not the atomic functions.  Define
   the atomic ones in terms of the sync ones.  */

extern void *backtrace_atomic_load_pointer (void *);
extern int backtrace_atomic_load_int (int *);
extern void backtrace_atomic_store_pointer (void *, void *);
extern void backtrace_atomic_store_size_t (size_t *, size_t);
extern void backtrace_atomic_store_int (int *, int);

#else /* !defined (HAVE_SYNC_FUNCTIONS) */

/* We have neither the sync nor the atomic functions.  These will
   never be called.  */

#define backtrace_atomic_load_pointer(p) (abort(), (void *) NULL)
#define backtrace_atomic_load_int(p) (abort(), 0)
#define backtrace_atomic_store_pointer(p, v) abort()
#define backtrace_atomic_store_size_t(p, v) abort()
#define backtrace_atomic_store_int(p, v) abort()

#endif /* !defined (HAVE_SYNC_FUNCTIONS) */
#endif /* !defined (HAVE_ATOMIC_FUNCTIONS) */

/* The type of the function that collects file/line information.  This
   is like backtrace_pcinfo.  */

typedef int (*fileline) (struct backtrace_state *state, uintptr_t pc,
			 backtrace_full_callback callback,
			 backtrace_error_callback error_callback, void *data);

/* The type of the function that collects symbol information.  This is
   like backtrace_syminfo.  */

typedef void (*syminfo) (struct backtrace_state *state, uintptr_t pc,
			 backtrace_syminfo_callback callback,
			 backtrace_error_callback error_callback, void *data);

/* What the backtrace state pointer points to.  */

struct backtrace_state
{
  /* The name of the executable.  */
  const char *filename;
  /* Non-zero if threaded.  */
  int threaded;
  /* The master lock for fileline_fn, fileline_data, syminfo_fn,
     syminfo_data, fileline_initialization_failed and everything the
     data pointers point to.  */
  void *lock;
  /* The function that returns file/line information.  */
  fileline fileline_fn;
  /* The data to pass to FILELINE_FN.  */
  void *fileline_data;
  /* The function that returns symbol information.  */
  syminfo syminfo_fn;
  /* The data to pass to SYMINFO_FN.  */
  void *syminfo_data;
  /* Whether initializing the file/line information failed.  */
  int fileline_initialization_failed;
  /* The lock for the freelist.  */
  int lock_alloc;
  /* The freelist when using mmap.  */
  struct backtrace_freelist_struct *freelist;
};

/* Open a file for reading.  Returns -1 on error.  If DOES_NOT_EXIST
   is not NULL, *DOES_NOT_EXIST will be set to 0 normally and set to 1
   if the file does not exist.  If the file does not exist and
   DOES_NOT_EXIST is not NULL, the function will return -1 and will
   not call ERROR_CALLBACK.  On other errors, or if DOES_NOT_EXIST is
   NULL, the function will call ERROR_CALLBACK before returning.  */
extern int backtrace_open (const char *filename,
			   backtrace_error_callback error_callback,
			   void *data,
			   int *does_not_exist);

/* A view of the contents of a file.  This supports mmap when
   available.  A view will remain in memory even after backtrace_close
   is called on the file descriptor from which the view was
   obtained.  */

struct backtrace_view
{
  /* The data that the caller requested.  */
  const void *data;
  /* The base of the view.  */
  void *base;
  /* The total length of the view.  */
  size_t len;
};

/* Create a view of SIZE bytes from DESCRIPTOR at OFFSET.  Store the
   result in *VIEW.  Returns 1 on success, 0 on error.  */
extern int backtrace_get_view (struct backtrace_state *state, int descriptor,
			       off_t offset, size_t size,
			       backtrace_error_callback error_callback,
			       void *data, struct backtrace_view *view);

/* Release a view created by backtrace_get_view.  */
extern void backtrace_release_view (struct backtrace_state *state,
				    struct backtrace_view *view,
				    backtrace_error_callback error_callback,
				    void *data);

/* Close a file opened by backtrace_open.  Returns 1 on success, 0 on
   error.  */

extern int backtrace_close (int descriptor,
			    backtrace_error_callback error_callback,
			    void *data);

/* Sort without using memory.  */

extern void backtrace_qsort (void *base, size_t count, size_t size,
			     int (*compar) (const void *, const void *));

/* Allocate memory.  This is like malloc.  If ERROR_CALLBACK is NULL,
   this does not report an error, it just returns NULL.  */

extern void *backtrace_alloc (struct backtrace_state *state, size_t size,
			      backtrace_error_callback error_callback,
			      void *data) ATTRIBUTE_MALLOC;

/* Free memory allocated by backtrace_alloc.  If ERROR_CALLBACK is
   NULL, this does not report an error.  */

extern void backtrace_free (struct backtrace_state *state, void *mem,
			    size_t size,
			    backtrace_error_callback error_callback,
			    void *data);

/* A growable vector of some struct.  This is used for more efficient
   allocation when we don't know the final size of some group of data
   that we want to represent as an array.  */

struct backtrace_vector
{
  /* The base of the vector.  */
  void *base;
  /* The number of bytes in the vector.  */
  size_t size;
  /* The number of bytes available at the current allocation.  */
  size_t alc;
};

/* Grow VEC by SIZE bytes.  Return a pointer to the newly allocated
   bytes.  Note that this may move the entire vector to a new memory
   location.  Returns NULL on failure.  */

extern void *backtrace_vector_grow (struct backtrace_state *state, size_t size,
				    backtrace_error_callback error_callback,
				    void *data,
				    struct backtrace_vector *vec);

/* Finish the current allocation on VEC.  Prepare to start a new
   allocation.  The finished allocation will never be freed.  Returns
   a pointer to the base of the finished entries, or NULL on
   failure.  */

extern void* backtrace_vector_finish (struct backtrace_state *state,
				      struct backtrace_vector *vec,
				      backtrace_error_callback error_callback,
				      void *data);

/* Release any extra space allocated for VEC.  This may change
   VEC->base.  Returns 1 on success, 0 on failure.  */

extern int backtrace_vector_release (struct backtrace_state *state,
				     struct backtrace_vector *vec,
				     backtrace_error_callback error_callback,
				     void *data);

/* Read initial debug data from a descriptor, and set the
   fileline_data, syminfo_fn, and syminfo_data fields of STATE.
   Return the fileln_fn field in *FILELN_FN--this is done this way so
   that the synchronization code is only implemented once.  This is
   called after the descriptor has first been opened.  It will close
   the descriptor if it is no longer needed.  Returns 1 on success, 0
   on error.  There will be multiple implementations of this function,
   for different file formats.  Each system will compile the
   appropriate one.  */

extern int backtrace_initialize (struct backtrace_state *state,
				 int descriptor,
				 backtrace_error_callback error_callback,
				 void *data,
				 fileline *fileline_fn);

/* Add file/line information for a DWARF module.  */

extern int backtrace_dwarf_add (struct backtrace_state *state,
				uintptr_t base_address,
				const unsigned char* dwarf_info,
				size_t dwarf_info_size,
				const unsigned char *dwarf_line,
				size_t dwarf_line_size,
				const unsigned char *dwarf_abbrev,
				size_t dwarf_abbrev_size,
				const unsigned char *dwarf_ranges,
				size_t dwarf_range_size,
				const unsigned char *dwarf_str,
				size_t dwarf_str_size,
				int is_bigendian,
				backtrace_error_callback error_callback,
				void *data, fileline *fileline_fn);

#endif
