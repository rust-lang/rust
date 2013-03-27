// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Helper functions used only in tests

#include "rust_sched_loop.h"
#include "rust_task.h"
#include "rust_util.h"
#include "rust_scheduler.h"
#include "sync/timer.h"
#include "sync/rust_thread.h"
#include "rust_abi.h"

// These functions are used in the unit tests for C ABI calls.

extern "C" CDECL uint32_t
rust_dbg_extern_identity_u32(uint32_t u) {
    return u;
}

extern "C" CDECL uint64_t
rust_dbg_extern_identity_u64(uint64_t u) {
    return u;
}

struct TwoU64s {
    uint64_t one;
    uint64_t two;
};

extern "C" CDECL TwoU64s
rust_dbg_extern_identity_TwoU64s(TwoU64s u) {
    return u;
}

struct TwoDoubles {
    double one;
    double two;
};

extern "C" CDECL TwoDoubles
rust_dbg_extern_identity_TwoDoubles(TwoDoubles u) {
    return u;
}

extern "C" CDECL double
rust_dbg_extern_identity_double(double u) {
    return u;
}

extern "C" CDECL char
rust_dbg_extern_identity_u8(char u) {
    return u;
}

extern "C" CDECL lock_and_signal *
rust_dbg_lock_create() {
    return new lock_and_signal();
}

extern "C" CDECL void
rust_dbg_lock_destroy(lock_and_signal *lock) {
    assert(lock);
    delete lock;
}

extern "C" CDECL void
rust_dbg_lock_lock(lock_and_signal *lock) {
    assert(lock);
    lock->lock();
}

extern "C" CDECL void
rust_dbg_lock_unlock(lock_and_signal *lock) {
    assert(lock);
    lock->unlock();
}

extern "C" CDECL void
rust_dbg_lock_wait(lock_and_signal *lock) {
    assert(lock);
    lock->wait();
}

extern "C" CDECL void
rust_dbg_lock_signal(lock_and_signal *lock) {
    assert(lock);
    lock->signal();
}

typedef void *(*dbg_callback)(void*);

extern "C" CDECL void *
rust_dbg_call(dbg_callback cb, void *data) {
    return cb(data);
}

extern "C" CDECL void rust_dbg_do_nothing() { }
