// Copyright (c) 2011 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef _RUST_ANDROID_DUMMY_H
#define _RUST_ANDROID_DUMMY_H

int backtrace (void **__array, int __size);

char **backtrace_symbols (void *__const *__array, int __size);

void backtrace_symbols_fd (void *__const *__array, int __size, int __fd);

#endif
