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

use cast;
use iter::Iterator;
use kinds::Send;
use mem;
use option::{Some, None};
use ptr::RawPtr;
use unstable::sync::Exclusive;
use slice::OwnedVector;
use vec::Vec;

type Queue = Exclusive<Vec<proc():Send>>;

// You'll note that these variables are *not* atomic, and this is done on
// purpose. This module is designed to have init() called *once* in a
// single-task context, and then run() is called only once in another
// single-task context. As a result of this, only the `push` function is
// thread-safe, and it assumes that the `init` function has run previously.
static mut QUEUE: *mut Queue = 0 as *mut Queue;
static mut RUNNING: bool = false;

pub fn init() {
    unsafe {
        rtassert!(!RUNNING);
        rtassert!(QUEUE.is_null());
        let state: ~Queue = box Exclusive::new(vec!());
        QUEUE = cast::transmute(state);
    }
}

pub fn push(f: proc():Send) {
    unsafe {
        rtassert!(!RUNNING);
        rtassert!(!QUEUE.is_null());
        let state: &mut Queue = cast::transmute(QUEUE);
        let mut f = Some(f);
        state.with(|arr|  {
            arr.push(f.take_unwrap());
        });
    }
}

pub fn run() {
    let vec = unsafe {
        rtassert!(!RUNNING);
        rtassert!(!QUEUE.is_null());
        RUNNING = true;
        let state: ~Queue = cast::transmute(QUEUE);
        QUEUE = 0 as *mut Queue;
        let mut vec = None;
        state.with(|arr| {
            vec = Some(mem::replace(arr, vec!()));
        });
        vec.take_unwrap()
    };


    for f in vec.move_iter() {
        f();
    }
}
