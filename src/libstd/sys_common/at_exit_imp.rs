// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of running at_exit routines
//!
//! Documentation can be found on the `rt::at_exit` function.

use alloc::boxed::FnBox;
use ptr;
use sys_common::mutex::Mutex;

type Queue = Vec<Box<FnBox()>>;

// NB these are specifically not types from `std::sync` as they currently rely
// on poisoning and this module needs to operate at a lower level than requiring
// the thread infrastructure to be in place (useful on the borders of
// initialization/destruction).
static LOCK: Mutex = Mutex::new();
static mut QUEUE: *mut Queue = ptr::null_mut();

// The maximum number of times the cleanup routines will be run. While running
// the at_exit closures new ones may be registered, and this count is the number
// of times the new closures will be allowed to register successfully. After
// this number of iterations all new registrations will return `false`.
const ITERS: usize = 10;

unsafe fn init() -> bool {
    if QUEUE.is_null() {
        let state: Box<Queue> = box Vec::new();
        QUEUE = Box::into_raw(state);
    } else if QUEUE as usize == 1 {
        // can't re-init after a cleanup
        return false
    }

    true
}

pub fn cleanup() {
    for i in 0..ITERS {
        unsafe {
            LOCK.lock();
            let queue = QUEUE;
            QUEUE = if i == ITERS - 1 {1} else {0} as *mut _;
            LOCK.unlock();

            // make sure we're not recursively cleaning up
            assert!(queue as usize != 1);

            // If we never called init, not need to cleanup!
            if queue as usize != 0 {
                let queue: Box<Queue> = Box::from_raw(queue);
                for to_run in *queue {
                    to_run();
                }
            }
        }
    }
}

pub fn push(f: Box<FnBox()>) -> bool {
    let mut ret = true;
    unsafe {
        LOCK.lock();
        if init() {
            (*QUEUE).push(f);
        } else {
            ret = false;
        }
        LOCK.unlock();
    }
    ret
}
