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

use core::prelude::*;

use boxed::Box;
use vec::Vec;
use mem;
use thunk::Thunk;
use sys_common::mutex::{Mutex, MUTEX_INIT};

type Queue = Vec<Thunk>;

// NB these are specifically not types from `std::sync` as they currently rely
// on poisoning and this module needs to operate at a lower level than requiring
// the thread infrastructure to be in place (useful on the borders of
// initialization/destruction).
static LOCK: Mutex = MUTEX_INIT;
static mut QUEUE: *mut Queue = 0 as *mut Queue;

unsafe fn init() {
    if QUEUE.is_null() {
        let state: Box<Queue> = box Vec::new();
        QUEUE = mem::transmute(state);
    } else {
        // can't re-init after a cleanup
        rtassert!(QUEUE as uint != 1);
    }

    // FIXME: switch this to use atexit as below. Currently this
    // segfaults (the queue's memory is mysteriously gone), so
    // instead the cleanup is tied to the `std::rt` entry point.
    //
    // ::libc::atexit(cleanup);
}

pub fn cleanup() {
    unsafe {
        LOCK.lock();
        let queue = QUEUE;
        QUEUE = 1 as *mut _;
        LOCK.unlock();

        // make sure we're not recursively cleaning up
        rtassert!(queue as uint != 1);

        // If we never called init, not need to cleanup!
        if queue as uint != 0 {
            let queue: Box<Queue> = mem::transmute(queue);
            for to_run in queue.into_iter() {
                to_run.invoke(());
            }
        }
    }
}

pub fn push(f: Thunk) {
    unsafe {
        LOCK.lock();
        init();
        (*QUEUE).push(f);
        LOCK.unlock();
    }
}
