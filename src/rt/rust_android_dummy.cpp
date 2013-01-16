// Copyright (c) 2011 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "rust_android_dummy.h"
#include <math.h>

#ifdef __ANDROID__

int backtrace(void **array, int size) { return 0; }

char **backtrace_symbols(void *const *array, int size) { return 0; }

void backtrace_symbols_fd (void *const *array, int size, int fd) {}


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
#endif
