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
use sync::{Mutex, atomic, Once, ONCE_INIT};
use mem;
use thunk::Thunk;

type Queue = Mutex<Vec<Thunk>>;

static INIT: Once = ONCE_INIT;
static QUEUE: atomic::AtomicUint = atomic::INIT_ATOMIC_UINT;

fn init() {
    let state: Box<Queue> = box Mutex::new(Vec::new());
    unsafe {
        QUEUE.store(mem::transmute(state), atomic::SeqCst);

        // FIXME: switch this to use atexit as below. Currently this
        // segfaults (the queue's memory is mysteriously gone), so
        // instead the cleanup is tied to the `std::rt` entry point.
        //
        // ::libc::atexit(cleanup);
    }
}

pub fn cleanup() {
    unsafe {
        let queue = QUEUE.swap(0, atomic::SeqCst);
        if queue != 0 {
            let queue: Box<Queue> = mem::transmute(queue);
            let v = mem::replace(&mut *queue.lock(), Vec::new());
            for to_run in v.into_iter() {
                to_run.invoke();
            }
        }
    }
}

pub fn push(f: Thunk) {
    INIT.doit(init);
    unsafe {
        // Note that the check against 0 for the queue pointer is not atomic at
        // all with respect to `run`, meaning that this could theoretically be a
        // use-after-free. There's not much we can do to protect against that,
        // however. Let's just assume a well-behaved runtime and go from there!
        let queue = QUEUE.load(atomic::SeqCst);
        rtassert!(queue != 0);
        (*(queue as *const Queue)).lock().push(f);
    }
}
