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

use boxed::FnBox;
use ptr;
use mem;
use sys_common::mutex::Mutex;

type Queue = Vec<Box<FnBox()>>;

// NB these are specifically not types from `std::sync` as they currently rely
// on poisoning and this module needs to operate at a lower level than requiring
// the thread infrastructure to be in place (useful on the borders of
// initialization/destruction).
static LOCK: Mutex = Mutex::new();
static mut QUEUE: *mut Queue = ptr::null_mut();

const DONE: *mut Queue = 1_usize as *mut _;

// The maximum number of times the cleanup routines will be run. While running
// the at_exit closures new ones may be registered, and this count is the number
// of times the new closures will be allowed to register successfully. After
// this number of iterations all new registrations will return `false`.
const ITERS: usize = 10;

unsafe fn init() -> bool {
    if QUEUE.is_null() {
        let state: Box<Queue> = box Vec::new();
        QUEUE = Box::into_raw(state);
    } else if QUEUE == DONE {
        // can't re-init after a cleanup
        return false
    }

    true
}

pub fn cleanup() {
    for i in 1..=ITERS {
        unsafe {
            let queue = {
                let _guard = LOCK.lock();
                mem::replace(&mut QUEUE, if i == ITERS { DONE } else { ptr::null_mut() })
            };

            // make sure we're not recursively cleaning up
            assert!(queue != DONE);

            // If we never called init, not need to cleanup!
            if !queue.is_null() {
                let queue: Box<Queue> = Box::from_raw(queue);
                for to_run in *queue {
                    to_run();
                }
            }
        }
    }
}

pub fn push(f: Box<FnBox()>) -> bool {
    unsafe {
        let _guard = LOCK.lock();
        if init() {
            (*QUEUE).push(f);
            true
        } else {
            false
        }
    }
}
