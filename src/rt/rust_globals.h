#ifndef RUST_GLOBALS_H
#define RUST_GLOBALS_H

#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS 1
#endif

#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS 1
#endif

#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS 1
#endif

#define ERROR 0

#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdarg.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <math.h>
#include <assert.h>

#include "rand.h"
#include "uthash.h"

#if defined(__WIN32__)
extern "C" {
#include <windows.h>
#include <tchar.h>
#include <wincrypt.h>
}
#elif defined(__GNUC__)
#include <unistd.h>
#include <dlfcn.h>
#include <pthread.h>
#include <errno.h>
#include <dirent.h>

#define GCC_VERSION (__GNUC__ * 10000 \
                     + __GNUC_MINOR__ * 100 \
                     + __GNUC_PATCHLEVEL__)

#else
#error "Platform not supported."
#endif

#ifdef __i386__
// 'cdecl' ABI only means anything on i386
#ifdef __WIN32__
#ifndef CDECL
#define CDECL __cdecl
#endif
#ifndef FASTCALL
#define FASTCALL __fastcall
#endif
#else
#define CDECL __attribute__((cdecl))
#define FASTCALL __attribute__((fastcall))
#endif
#else
#define CDECL
#define FASTCALL
#endif

/* Controls whether claims are turned into checks */
/* Variable name must be kept consistent with trans.rs */
extern "C" int check_claims;

#define CHECKED(call)                                               \
    {                                                               \
    int res = (call);                                               \
        if(0 != res) {                                              \
            fprintf(stderr,                                         \
                    #call " failed in %s at line %d, result = %d "  \
                    "(%s) \n",                                      \
                    __FILE__, __LINE__, res, strerror(res));        \
            abort();                                                \
        }                                                           \
    }

#define MUST_CHECK __attribute__((warn_unused_result))

#define PTR "0x%" PRIxPTR

// This accounts for logging buffers.
static size_t const BUF_BYTES = 2048;

#define INIT_TASK_ID 1

// The error status to use when the process fails
#define PROC_FAIL_CODE 101

// A cond(ition) is something we can block on. This can be a channel
// (writing), a port (reading) or a task (waiting).
struct rust_cond { };

#endif /* RUST_GLOBALS_H */
