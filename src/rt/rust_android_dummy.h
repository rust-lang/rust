// Copyright (c) 2011 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef _RUST_ANDROID_DUMMY_H
#define _RUST_ANDROID_DUMMY_H

int backtrace (void **__array, int __size);

char **backtrace_symbols (void *__const *__array, int __size);

void backtrace_symbols_fd (void *__const *__array, int __size, int __fd);

#include <sys/types.h>

struct stat;
typedef struct {
    size_t gl_pathc;    /* Count of total paths so far. */
    size_t gl_matchc;   /* Count of paths matching pattern. */
    size_t gl_offs;     /* Reserved at beginning of gl_pathv. */
    int gl_flags;       /* Copy of flags parameter to glob. */
    char **gl_pathv;    /* List of paths matching pattern. */
                /* Copy of errfunc parameter to glob. */
    int (*gl_errfunc)(const char *, int);

    /*
     * Alternate filesystem access methods for glob; replacement
     * versions of closedir(3), readdir(3), opendir(3), stat(2)
     * and lstat(2).
     */
    void (*gl_closedir)(void *);
    struct dirent *(*gl_readdir)(void *);
    void *(*gl_opendir)(const char *);
    int (*gl_lstat)(const char *, struct stat *);
} glob_t;

#endif
