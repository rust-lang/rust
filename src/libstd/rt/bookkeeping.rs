// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Task bookkeeping
//!
//! This module keeps track of the number of running tasks so that entry points
//! with libnative know when it's possible to exit the program (once all tasks
//! have exited).
//!
//! The green counterpart for this is bookkeeping on sched pools, and it's up to
//! each respective runtime to make sure that they call increment() and
//! decrement() manually.

#[experimental]; // this is a massive code smell
#[doc(hidden)];

use sync::atomics;
use unstable::mutex::{StaticNativeMutex, NATIVE_MUTEX_INIT};

static mut TASK_COUNT: atomics::AtomicUint = atomics::INIT_ATOMIC_UINT;
static mut TASK_LOCK: StaticNativeMutex = NATIVE_MUTEX_INIT;

pub fn increment() {
    let _ = unsafe { TASK_COUNT.fetch_add(1, atomics::SeqCst) };
}

pub fn decrement() {
    unsafe {
        if TASK_COUNT.fetch_sub(1, atomics::SeqCst) == 1 {
            let guard = TASK_LOCK.lock();
            guard.signal();
        }
    }
}

/// Waits for all other native tasks in the system to exit. This is only used by
/// the entry points of native programs
pub fn wait_for_other_tasks() {
    unsafe {
        let guard = TASK_LOCK.lock();
        while TASK_COUNT.load(atomics::SeqCst) > 0 {
            guard.wait();
        }
    }
}
