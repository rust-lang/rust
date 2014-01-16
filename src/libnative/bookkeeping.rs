// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! 1:1 Task bookkeeping
//!
//! This module keeps track of the number of running 1:1 tasks so that entry
//! points with libnative know when it's possible to exit the program (once all
//! tasks have exited).
//!
//! The green counterpart for this is bookkeeping on sched pools.

use std::sync::atomics;
use std::unstable::mutex::{Mutex, MUTEX_INIT};

static mut TASK_COUNT: atomics::AtomicUint = atomics::INIT_ATOMIC_UINT;
static mut TASK_LOCK: Mutex = MUTEX_INIT;

pub fn increment() {
    unsafe { TASK_COUNT.fetch_add(1, atomics::SeqCst); }
}

pub fn decrement() {
    unsafe {
        if TASK_COUNT.fetch_sub(1, atomics::SeqCst) == 1 {
            TASK_LOCK.lock();
            TASK_LOCK.signal();
            TASK_LOCK.unlock();
        }
    }
}

/// Waits for all other native tasks in the system to exit. This is only used by
/// the entry points of native programs
pub fn wait_for_other_tasks() {
    unsafe {
        TASK_LOCK.lock();
        while TASK_COUNT.load(atomics::SeqCst) > 0 {
            TASK_LOCK.wait();
        }
        TASK_LOCK.unlock();
    }
}
