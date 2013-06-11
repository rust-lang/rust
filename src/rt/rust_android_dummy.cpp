// Copyright (c) 2011 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifdef __ANDROID__

#include "rust_android_dummy.h"
#include <math.h>
#include <errno.h>

int backtrace(void **array, int size) { return 0; }

char **backtrace_symbols(void *const *array, int size) { return 0; }

void backtrace_symbols_fd (void *const *array, int size, int fd) {}

extern "C" volatile int* __errno_location() {
    return &errno;
}

extern "C" float log2f(float f)
{
    return logf( f ) / logf( 2 );
}

extern "C" double log2( double n )
{
    return log( n ) / log( 2 );
}

extern "C" void telldir()
{
}

extern "C" void seekdir()
{
}

extern "C" void mkfifo()
{
}

extern "C" void abs()
{
}

extern "C" void labs()
{
}

extern "C" void rand()
{
}

extern "C" void srand()
{
}

extern "C" void atof()
{
}

extern "C" void tgammaf()
{
}

extern "C" int glob(const char *pattern,
                    int flags,
                    int (*errfunc) (const char *epath, int eerrno),
                    glob_t *pglob)
{
    return 0;
}

extern "C" void globfree(glob_t *pglob)
{
}

extern "C" int pthread_atfork(void (*prefork)(void),
                              void (*postfork_parent)(void),
                              void (*postfork_child)(void))
{
    return 0;
}

#endif
