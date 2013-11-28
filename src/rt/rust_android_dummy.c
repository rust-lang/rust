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

volatile int* __errno_location() {
    return &errno;
}

float log2f(float f)
{
    return logf( f ) / logf( 2 );
}

double log2( double n )
{
    return log( n ) / log( 2 );
}

void telldir()
{
}

void seekdir()
{
}

void mkfifo()
{
}

void abs()
{
}

void labs()
{
}

void rand()
{
}

void srand()
{
}

void atof()
{
}

int glob(const char *pattern,
                    int flags,
                    int (*errfunc) (const char *epath, int eerrno),
                    glob_t *pglob)
{
    return 0;
}

void globfree(glob_t *pglob)
{
}

int pthread_atfork(void (*prefork)(void),
                              void (*postfork_parent)(void),
                              void (*postfork_child)(void))
{
    return 0;
}

int mlockall(int flags)
{
    return 0;
}

int munlockall(void)
{
    return 0;
}

int shm_open(const char *name, int oflag, mode_t mode)
{
    return 0;
}

int shm_unlink(const char *name)
{
    return 0;
}

int posix_madvise(void *addr, size_t len, int advice)
{
    return 0;
}

#endif
