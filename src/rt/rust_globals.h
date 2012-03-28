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

#include "rust.h"
#include "rand.h"
#include "uthash.h"
#include "rust_env.h"

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

#endif /* RUST_GLOBALS_H */
