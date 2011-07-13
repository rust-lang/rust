#ifndef GLOBALS_H
#define GLOBALS_H

#ifndef RUST_INTERNAL_H
// these are defined in two files and GCC complains
#define __STDC_LIMIT_MACROS 1
#define __STDC_CONSTANT_MACROS 1
#define __STDC_FORMAT_MACROS 1
#endif

#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

#include <stdio.h>
#include <string.h>

#if defined(__WIN32__)
extern "C" {
#include <windows.h>
#include <tchar.h>
#include <wincrypt.h>
}
#elif defined(__GNUC__)
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <pthread.h>
#include <errno.h>
#include <time.h>
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

#endif /* GLOBALS_H */
