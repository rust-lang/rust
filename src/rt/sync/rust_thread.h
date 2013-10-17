// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#ifndef RUST_THREAD_H
#define RUST_THREAD_H

#include "rust_globals.h"

/**
 * Thread utility class. Derive and implement your own run() method.
 */
class rust_thread {
 private:
#if defined(__WIN32__)
    HANDLE thread;
#else
    pthread_t thread;
#endif
 public:

    rust_thread();
    virtual ~rust_thread();

    void start();

    virtual void run() = 0;

    void join();
    void detach();
};

#endif /* RUST_THREAD_H */
