// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#include "rust_thread.h"
#include <limits.h>

const size_t default_stack_sz = 1024*1024;

rust_thread::rust_thread() : thread(0) {
}

rust_thread::~rust_thread() {
}

#if defined(__WIN32__)
static DWORD WINAPI
#elif defined(__GNUC__)
static void *
#else
#error "Platform not supported"
#endif
rust_thread_start(void *ptr) {
    rust_thread *thread = (rust_thread *) ptr;
    thread->run();
    return 0;
}

void
rust_thread::start() {
#if defined(__WIN32__)
   thread = CreateThread(NULL, default_stack_sz, rust_thread_start, this, 0, NULL);
#else
   // PTHREAD_STACK_MIN of some system is larger than default size
   // so we check stack_sz to prevent assertion failure.
   size_t stack_sz = default_stack_sz;
   if (stack_sz < PTHREAD_STACK_MIN) {
      stack_sz = PTHREAD_STACK_MIN;
   }
   pthread_attr_t attr;
   CHECKED(pthread_attr_init(&attr));
   CHECKED(pthread_attr_setstacksize(&attr, stack_sz));
   CHECKED(pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE));
   CHECKED(pthread_create(&thread, &attr, rust_thread_start, (void *) this));
#endif
}

void
rust_thread::join() {
#if defined(__WIN32__)
   if (thread)
     WaitForSingleObject(thread, INFINITE);
#else
   if (thread)
     CHECKED(pthread_join(thread, NULL));
#endif
   thread = 0;
}

void
rust_thread::detach() {
#if !defined(__WIN32__)
    // Don't leak pthread resources.
    // http://crosstantine.blogspot.com/2010/01/pthreadcreate-memory-leak.html
    CHECKED(pthread_detach(thread));
#endif
}
